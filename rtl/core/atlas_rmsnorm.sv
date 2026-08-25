//===========================================================================
// atlas_rmsnorm -- RMS normalisation over one hidden row
//
//     y_i = x_i * w_i / sqrt( (1/N) * sum_j x_j^2  +  eps )
//
// Qwen3 uses RMSNorm with eps 1e-6 (rms_norm_eps in the model config), twice
// per layer plus once at the end, over rows of 4096.
//
// Two passes, because the scale depends on the whole row: the caller streams
// the row once to accumulate the sum of squares, then streams it again to
// scale.  Re-reading from the core's scratchpad costs far less than buffering
// 4096 binary32 values inside the normaliser.
//
// The sum of squares uses PARTS partial accumulators rather than one.  The
// binary32 adder takes several cycles, so a single running total would be read
// stale by every add issued inside that window -- the same recurrence the PE's
// accumulator bank exists to break.  Rotating over PARTS partials means one is
// only re-read after its previous update has landed, and they are folded
// together once the row ends.
//
// 1/N is supplied by the caller rather than divided for here: row length is a
// property of the model and constant for a whole inference run.
//===========================================================================
`include "atlas_defs.svh"

module atlas_rmsnorm #(
  parameter int unsigned PARTS   = 4,     // must exceed ADD_LAT
  parameter int unsigned CNT_W   = 16
) (
  input  logic        clk,
  input  logic        rst_n,

  input  logic        start,        // begin a new row: clears the accumulator
  input  logic [31:0] inv_n,        // 1/N as binary32
  input  logic [31:0] eps,          // rms_norm_eps as binary32

  // ---- pass 1: accumulate sum of squares ----
  input  logic        acc_valid,
  input  logic [31:0] acc_x,
  input  logic        acc_last,     // final element of the row
  output logic        scale_ready,
  output logic [31:0] scale,        // 1/sqrt(mean + eps)

  // ---- pass 2: scale ----
  input  logic        mul_valid,
  input  logic [31:0] mul_x,
  input  logic [31:0] mul_w,
  output logic        out_valid,
  output logic [31:0] out_y
);

  localparam int unsigned ADD_LAT = 2;
  localparam int unsigned PW      = $clog2(PARTS);

  typedef enum logic [2:0] {
    R_IDLE, R_ACC, R_DRAIN, R_REDUCE, R_MEAN, R_EPS, R_RSQRT, R_READY
  } state_e;

  state_e state;

  logic [31:0]      part [0:PARTS-1];
  logic [PW-1:0]    sel;

  // Which partial each in-flight add writes back to.  Kept in a FIFO rather
  // than a fixed delay chain: adds complete in issue order, but the number of
  // cycles between issuing one and its result is a property of the adder plus
  // the register on its valid, and a delay chain sized to the wrong number
  // silently writes results into the wrong partial.
  localparam int unsigned WBD = 8;
  logic [PW-1:0]    wb_fifo [0:WBD-1];
  logic [$clog2(WBD):0] wb_wr, wb_rd;
  logic [CNT_W-1:0] n_issued, n_added;
  logic             last_seen;
  logic [PW:0]      red_i;
  logic [31:0]      red_acc;
  logic [3:0]       wait_ct;

  //-------------------------------------------------------------------------
  // Shared arithmetic for pass 1 and the scale computation
  //-------------------------------------------------------------------------
  logic        sq_in, sq_v;
  logic [31:0] sq_a, sq_b, sq_y;
  atlas_fp32_mul u_sq (.clk(clk), .rst_n(rst_n), .in_valid(sq_in),
                       .a(sq_a), .b(sq_b), .out_valid(sq_v), .result(sq_y));

  logic        ad_in, ad_v;
  logic [31:0] ad_a, ad_b, ad_y;
  atlas_fp32_add u_ad (.clk(clk), .rst_n(rst_n), .in_valid(ad_in),
                       .a(ad_a), .b(ad_b), .out_valid(ad_v), .result(ad_y));

  logic        rq_in, rq_v;
  logic [31:0] rq_x, rq_y;
  atlas_rsqrt_f32 u_rq (.clk(clk), .rst_n(rst_n), .in_valid(rq_in),
                        .x(rq_x), .out_valid(rq_v), .y(rq_y));

  //-------------------------------------------------------------------------
  // Pass 2 is pure dataflow: (x * w) * scale, one element per cycle.
  //-------------------------------------------------------------------------
  logic        m1_v;
  logic [31:0] m1_y;
  atlas_fp32_mul u_m1 (.clk(clk), .rst_n(rst_n), .in_valid(mul_valid),
                       .a(mul_x), .b(mul_w), .out_valid(m1_v), .result(m1_y));

  atlas_fp32_mul u_m2 (.clk(clk), .rst_n(rst_n), .in_valid(m1_v),
                       .a(m1_y), .b(scale), .out_valid(out_valid), .result(out_y));

  integer k;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= R_IDLE;
      scale       <= 32'h3F800000;
      scale_ready <= 1'b0;
      sq_in <= 1'b0; ad_in <= 1'b0; rq_in <= 1'b0;
      n_issued <= '0; n_added <= '0; last_seen <= 1'b0;
      sel <= '0; red_i <= '0; wait_ct <= 4'd0;
      wb_wr <= '0; wb_rd <= '0;
      for (k = 0; k < PARTS; k = k + 1) part[k] <= 32'h0;
    end else begin
      sq_in <= 1'b0; ad_in <= 1'b0; rq_in <= 1'b0;

      case (state)
        //-----------------------------------------------------------------
        R_IDLE, R_READY: begin
          if (start) begin
            for (k = 0; k < PARTS; k = k + 1) part[k] <= 32'h0;
            n_issued    <= '0;
            n_added     <= '0;
            last_seen   <= 1'b0;
            sel         <= '0;
            wb_wr       <= '0;
            wb_rd       <= '0;
            scale_ready <= 1'b0;
            state       <= R_ACC;
          end
        end

        //-----------------------------------------------------------------
        // Square every element and fold it into a rotating partial sum.
        //-----------------------------------------------------------------
        R_ACC, R_DRAIN: begin
          if ((state == R_ACC) && acc_valid) begin
            sq_a     <= acc_x;
            sq_b     <= acc_x;
            sq_in    <= 1'b1;
            n_issued <= n_issued + 1'b1;
            if (acc_last) begin
              last_seen <= 1'b1;
              state     <= R_DRAIN;
            end
          end

          if (sq_v) begin
            ad_a  <= part[sel];
            ad_b  <= sq_y;
            ad_in <= 1'b1;
            wb_fifo[wb_wr[$clog2(WBD)-1:0]] <= sel;
            wb_wr <= wb_wr + 1'b1;
            sel   <= (sel == PW'(PARTS-1)) ? '0 : (sel + 1'b1);
          end

          if (ad_v) begin
            part[wb_fifo[wb_rd[$clog2(WBD)-1:0]]] <= ad_y;
            wb_rd   <= wb_rd + 1'b1;
            n_added <= n_added + 1'b1;
            // The row is complete only once every square issued has been
            // folded in, not merely when the last one arrived at the input.
            if (last_seen && ((n_added + 1'b1) == n_issued)) begin
              red_i   <= '0;
              red_acc <= 32'h0;
              wait_ct <= 4'd0;
              state   <= R_REDUCE;
            end
          end
        end

        //-----------------------------------------------------------------
        R_REDUCE: begin
          if (wait_ct == 4'd0) begin
            ad_a    <= red_acc;
            ad_b    <= part[red_i[PW-1:0]];
            ad_in   <= 1'b1;
            wait_ct <= 4'd1;
          end else if (ad_v) begin
            red_acc <= ad_y;
            wait_ct <= 4'd0;
            if (red_i == (PW+1)'(PARTS-1)) state <= R_MEAN;
            else                           red_i <= red_i + 1'b1;
          end
        end

        //-----------------------------------------------------------------
        R_MEAN: begin
          if (wait_ct == 4'd0) begin
            sq_a    <= red_acc;
            sq_b    <= inv_n;
            sq_in   <= 1'b1;
            wait_ct <= 4'd1;
          end else if (sq_v) begin
            ad_a    <= sq_y;
            ad_b    <= eps;
            ad_in   <= 1'b1;
            wait_ct <= 4'd0;
            state   <= R_EPS;
          end
        end

        R_EPS: begin
          if (ad_v) begin
            rq_x  <= ad_y;
            rq_in <= 1'b1;
            state <= R_RSQRT;
          end
        end

        R_RSQRT: begin
          if (rq_v) begin
            scale       <= rq_y;
            scale_ready <= 1'b1;
            state       <= R_READY;
          end
        end

        default: state <= R_IDLE;
      endcase
    end
  end

endmodule
