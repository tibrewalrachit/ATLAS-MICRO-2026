//===========================================================================
// atlas_moe_router -- top-k expert selection for Qwen3-235B-A22B
//
// Qwen3-235B-A22B has 128 experts per layer and routes each token to 8 of
// them (configs/models/qwen3_235b_a22b.json: num_experts 128,
// num_experts_per_tok 8, norm_topk_prob true).  Reference behaviour is
//
//     w = softmax(logits)          over all 128
//     w, idx = topk(w, 8)
//     w = w / sum(w)               because norm_topk_prob is set
//
// The 128-way softmax is never computed here, and it does not need to be.
// Softmax is monotonic, so the top 8 of softmax(logits) are the top 8 of
// logits; and the global denominator that softmax would divide by is a common
// factor of all 8 selected weights, which the renormalisation then divides
// out again.  So
//
//     w_j = exp(l_j - max) / sum_k exp(l_k - max)   over the selected 8 only
//
// is exactly equal to the reference, with 8 exponentials instead of 128 and
// no 128-wide reduction.  Subtracting the max is free -- it is already the
// first selected logit -- and keeps every exponential at or below 1.
//
// Logits stream in one per cycle; a token's routing costs roughly 200 cycles
// against the tens of thousands its expert GEMMs take, so the arithmetic is
// sequenced through one adder, one exponential, one reciprocal and one
// multiplier rather than replicated.
//===========================================================================
`include "atlas_defs.svh"

module atlas_moe_router #(
  parameter int unsigned NEXP  = QWEN_EXPERTS,   // 128
  parameter int unsigned TOPK  = QWEN_TOPK,      // 8
  parameter int unsigned IDX_W = $clog2(NEXP),
  parameter int unsigned KW    = $clog2(TOPK)      // width of a top-k slot index
) (
  input  logic                    clk,
  input  logic                    rst_n,

  input  logic                    start,       // pulse before streaming logits
  input  logic                    lg_valid,    // one router logit per cycle
  input  logic [31:0]             lg_data,

  output logic                    done,
  output logic [TOPK*IDX_W-1:0]   sel_idx,     // chosen experts, best first
  output logic [TOPK*32-1:0]      sel_w        // normalised weights, binary32
);

  //-------------------------------------------------------------------------
  // Ordering binary32 values by their bit pattern.
  //
  // For non-negative floats the unsigned bit pattern already sorts correctly.
  // For negative ones the order is reversed.  Flipping all bits of a negative
  // and only the sign of a non-negative maps both onto one unsigned key whose
  // order matches the numeric order, so the comparators below are plain
  // unsigned ones.
  //-------------------------------------------------------------------------
  function automatic logic [31:0] fkey(input logic [31:0] v);
    fkey = v[31] ? ~v : (v | 32'h8000_0000);
  endfunction

  typedef enum logic [2:0] {
    S_IDLE, S_LOAD, S_EXP, S_SUM, S_RECIP, S_SCALE, S_DONE
  } state_e;

  state_e state;

  logic [31:0]      top_key [0:TOPK-1];
  logic [31:0]      top_val [0:TOPK-1];
  logic [IDX_W-1:0] top_idx [0:TOPK-1];
  logic [31:0]      expv    [0:TOPK-1];

  logic [IDX_W:0]   lg_count;
  logic [KW-1:0]    j;          // which of the TOPK is being processed
  logic [5:0]       wait_ct;
  logic [31:0]      sum_acc, recip_v;

  //-------------------------------------------------------------------------
  // Arithmetic resources, shared across the phases
  //-------------------------------------------------------------------------
  logic        add_in, add_out;
  logic [31:0] add_a, add_b, add_r;
  atlas_fp32_add u_add (.clk(clk), .rst_n(rst_n), .in_valid(add_in),
                        .a(add_a), .b(add_b), .out_valid(add_out), .result(add_r));

  logic        exp_in, exp_out;
  logic [31:0] exp_x, exp_y;
  atlas_exp_f32 u_exp (.clk(clk), .rst_n(rst_n), .in_valid(exp_in),
                       .x(exp_x), .out_valid(exp_out), .y(exp_y));

  logic        rcp_in, rcp_out;
  logic [31:0] rcp_x, rcp_y;
  atlas_recip_f32 u_rcp (.clk(clk), .rst_n(rst_n), .in_valid(rcp_in),
                         .x(rcp_x), .out_valid(rcp_out), .y(rcp_y));

  logic        mul_in, mul_out;
  logic [31:0] mul_a, mul_b, mul_r;
  atlas_fp32_mul u_mul (.clk(clk), .rst_n(rst_n), .in_valid(mul_in),
                        .a(mul_a), .b(mul_b), .out_valid(mul_out), .result(mul_r));

  //-------------------------------------------------------------------------
  // Top-k insertion.  top_key is kept in descending order, so the comparison
  // vector against a new logit is a run of zeros followed by a run of ones and
  // the first set bit is the insertion point.
  //-------------------------------------------------------------------------
  logic [31:0]   new_key;
  logic [TOPK-1:0] gt;
  logic [KW-1:0] ins_pos;
  logic          ins_any;

  integer g;
  always_comb begin
    new_key = fkey(lg_data);
    for (g = 0; g < TOPK; g = g + 1)
      gt[g] = (new_key > top_key[g]);
    ins_any = |gt;
    ins_pos = '0;
    for (g = TOPK-1; g >= 0; g = g - 1)
      if (gt[g]) ins_pos = KW'(g);
  end

  integer k;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      done     <= 1'b0;
      lg_count <= '0;
      j        <= '0;
      wait_ct  <= 6'd0;
      sum_acc  <= 32'h0;
      for (k = 0; k < TOPK; k = k + 1) begin
        top_key[k] <= 32'h0;      // below every real key
        top_val[k] <= 32'h0;
        top_idx[k] <= '0;
      end
      add_in <= 1'b0; exp_in <= 1'b0; rcp_in <= 1'b0; mul_in <= 1'b0;
    end else begin
      add_in <= 1'b0; exp_in <= 1'b0; rcp_in <= 1'b0; mul_in <= 1'b0;
      done   <= 1'b0;

      case (state)
        //-------------------------------------------------------------
        S_IDLE: begin
          if (start) begin
            lg_count <= '0;
            sum_acc  <= 32'h0;
            for (k = 0; k < TOPK; k = k + 1) begin
              top_key[k] <= 32'h0;
              top_val[k] <= 32'h0;
              top_idx[k] <= '0;
            end
            state <= S_LOAD;
          end
        end

        //-------------------------------------------------------------
        S_LOAD: begin
          if (lg_valid) begin
            if (ins_any) begin
              for (k = 0; k < TOPK; k = k + 1) begin
                if (KW'(k) == ins_pos) begin
                  top_key[k] <= new_key;
                  top_val[k] <= lg_data;
                  top_idx[k] <= lg_count[IDX_W-1:0];
                end else if (KW'(k) > ins_pos) begin
                  top_key[k] <= top_key[k-1];
                  top_val[k] <= top_val[k-1];
                  top_idx[k] <= top_idx[k-1];
                end
              end
            end
            lg_count <= lg_count + 1'b1;
            if (lg_count == (IDX_W+1)'(NEXP-1)) begin
              j       <= '0;
              wait_ct <= 6'd0;
              state   <= S_EXP;
            end
          end
        end

        //-------------------------------------------------------------
        // exp(l_j - max).  top_val[0] is the max, so the subtraction is
        // an add against its sign flip.
        //-------------------------------------------------------------
        S_EXP: begin
          if (wait_ct == 6'd0) begin
            add_a   <= top_val[j];
            add_b   <= {~top_val[0][31], top_val[0][30:0]};
            add_in  <= 1'b1;
            wait_ct <= 6'd1;
          end else if (add_out) begin
            exp_x  <= add_r;
            exp_in <= 1'b1;
          end else if (exp_out) begin
            expv[j] <= exp_y;
            if (j == KW'(TOPK-1)) begin
              j       <= '0;
              wait_ct <= 6'd0;
              sum_acc <= 32'h0;
              state   <= S_SUM;
            end else begin
              j       <= j + 1'b1;
              wait_ct <= 6'd0;
            end
          end
        end

        //-------------------------------------------------------------
        S_SUM: begin
          if (wait_ct == 6'd0) begin
            add_a   <= sum_acc;
            add_b   <= expv[j];
            add_in  <= 1'b1;
            wait_ct <= 6'd1;
          end else if (add_out) begin
            sum_acc <= add_r;
            wait_ct <= 6'd0;
            if (j == KW'(TOPK-1)) begin
              j     <= '0;
              state <= S_RECIP;
            end else begin
              j <= j + 1'b1;
            end
          end
        end

        //-------------------------------------------------------------
        S_RECIP: begin
          if (wait_ct == 6'd0) begin
            rcp_x   <= sum_acc;
            rcp_in  <= 1'b1;
            wait_ct <= 6'd1;
          end else if (rcp_out) begin
            recip_v <= rcp_y;
            wait_ct <= 6'd0;
            j       <= '0;
            state   <= S_SCALE;
          end
        end

        //-------------------------------------------------------------
        S_SCALE: begin
          if (wait_ct == 6'd0) begin
            mul_a   <= expv[j];
            mul_b   <= recip_v;
            mul_in  <= 1'b1;
            wait_ct <= 6'd1;
          end else if (mul_out) begin
            sel_w[j*32 +: 32] <= mul_r;
            wait_ct <= 6'd0;
            if (j == KW'(TOPK-1)) state <= S_DONE;
            else                 j <= j + 1'b1;
          end
        end

        //-------------------------------------------------------------
        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  genvar gi;
  generate
    for (gi = 0; gi < TOPK; gi++) begin : g_idx
      assign sel_idx[gi*IDX_W +: IDX_W] = top_idx[gi];
    end
  endgenerate

endmodule
