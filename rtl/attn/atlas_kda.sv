//===========================================================================
// atlas_kda -- Kimi Delta Attention state engine
//
// KDA replaces a growing KV cache with a fixed-size recurrent state.  The
// published recurrence is
//
//     S_t = diag(alpha_t) . S_{t-1} - beta_t k_t k_t^T S_{t-1} + beta_t k_t v_t^T
//
// with alpha_t (decay) and beta_t (update) both produced by a 2-tap
// convolution over k_t.
//
// Forming k k^T S as written would be d^3 work per token.  The last two terms
// share the k_t outer product, so the engine computes instead
//
//     u_t = v_t - S_{t-1}^T k_t                    one matrix-vector product
//     S_t = diag(alpha_t) . S_{t-1} + beta_t k_t u_t^T      one rank-1 update
//
// which is d^2.  Elementwise, and noting diag(alpha) multiplies from the left
// so alpha scales *rows*:
//
//     u[j]    = v[j] - sum_i k[i] * S[i][j]
//     S[i][j] = alpha[i] * S[i][j] + (beta * k[i]) * u[j]
//
// Why this matters for a memory-bound accelerator: the state is d x d whatever
// the context length.  Attention stops being a bandwidth problem that grows
// with the sequence and becomes a fixed working set, which for a design whose
// whole premise is bonded DRAM bandwidth changes what the memory system has to
// carry.  docs/rtl/README.md works the numbers for Qwen-scale models.
//
// One row of the state is processed per cycle, D lanes wide.  Both passes and
// the query read-out share one multiply/add datapath per lane.
//
// The state is held in registers here so the whole engine stays inside
// standard-cell synthesis and static timing.  A real d = 128 head is 64 KB per
// head and belongs in a macro; D is a parameter for exactly that reason.
//===========================================================================
`include "atlas_defs.svh"

module atlas_kda #(
  parameter int unsigned D     = 16,   // head dimension (128 in the model)
  // Partial accumulators per lane.  The multiply/add chain is several cycles
  // deep and a row is issued every cycle, so a single running total would be
  // read stale -- the same recurrence atlas_pe and atlas_rmsnorm break the
  // same way.  PARTS must exceed that depth.
  parameter int unsigned PARTS = 8,
  parameter int unsigned PW    = $clog2(PARTS),
  parameter int unsigned IW    = $clog2(D) + 1
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              start,        // begin one token step
  input  logic [D*32-1:0]   k_vec,
  input  logic [D*32-1:0]   v_vec,
  input  logic [D*32-1:0]   alpha_vec,    // decay gate, per row
  input  logic [31:0]       beta,         // update gate
  input  logic [D*32-1:0]   q_vec,

  // State access.  A recurrent state is what replaces the KV cache, so it has
  // to be saveable and restorable the same way a KV cache is paged: one row
  // per access, usable while the engine is idle.
  input  logic              st_we,
  input  logic [IW-1:0]     st_row,
  input  logic [D*32-1:0]   st_wdata,
  output logic [D*32-1:0]   st_rdata,

  output logic              busy,
  output logic              done,
  output logic [D*32-1:0]   o_vec         // read-out, S_t^T q_t
);

  //-------------------------------------------------------------------------
  // State, held row-major: S[i][j] at ((i*D)+j)*32.
  //-------------------------------------------------------------------------
  logic [D*D*32-1:0] S;

  typedef enum logic [3:0] {
    K_IDLE, K_MV, K_MV_DRAIN, K_FOLD, K_SUB, K_BK,
    K_UPD, K_UPD_DRAIN, K_RD, K_RD_DRAIN, K_RD_FOLD, K_DONE
  } state_e;

  state_e st;

  logic [IW-1:0]  row;          // row being issued
  logic [IW-1:0]  n_done;       // rows retired
  logic [PW-1:0]  sel;          // rotating partial
  logic [PW:0]    fold_i;
  logic           fold_busy;
  logic           sub_busy;
  logic           second_pass;  // K_MV is reused for the read-out

  //-------------------------------------------------------------------------
  // Per-lane datapath: two multipliers and one adder.
  //   pass 1 / read-out : mulA = vec[row] * S[row][j], adder accumulates
  //   update            : mulA = alpha[row] * S[row][j]
  //                       mulB = (beta*k[row]) * u[j], adder sums them
  //-------------------------------------------------------------------------
  logic [D*32-1:0] mulA_y, mulB_y, add_y;
  logic [D-1:0]    mulA_vv, mulB_vv, add_vv;
  logic            mulA_v, mulB_v, add_v;

  logic [31:0]     scal_a;      // broadcast scalar for mulA
  logic [31:0]     scal_b;      // broadcast scalar for mulB
  logic            mul_in, add_in_mv, add_in_upd;

  logic [D*32-1:0] row_data;
  assign row_data = S[32'(row)*D*32 +: D*32];

  // Operands are captured into registers alongside the valid they travel with.
  // The valid is registered, so the arithmetic units see it one cycle after it
  // is set; a combinational operand would by then have moved on to the next
  // row or the next partial, and the unit would compute on the wrong pair.
  logic [D*32-1:0] rowd_q, add_a_q, add_b_q, mulb_q;

  // beta*k, computed for every row at once before the update pass.  Producing
  // it per row inside the update would need the scalar a cycle before the
  // multiply that consumes it, and a scalar arriving late is silent: the
  // update simply uses the previous row's gate.
  logic [D*32-1:0] bk_vec;
  logic            bk_busy;

  // Partial accumulators, and the folded result.
  logic [D*PARTS*32-1:0] part;
  logic [D*32-1:0]       u_vec;

  // Write-back index for in-flight accumulate adds.  A FIFO rather than a
  // fixed delay chain: the depth of multiply-then-add is a property of those
  // units, and a chain sized to the wrong number writes into the wrong partial.
  localparam int unsigned WBD = 16;
  logic [PW-1:0]           wb [0:WBD-1];
  logic [$clog2(WBD):0]    wb_wr, wb_rd;
  // Row index for in-flight update adds, same reasoning.
  logic [IW-1:0]           rb [0:WBD-1];
  logic [$clog2(WBD):0]    rb_wr, rb_rd;

  logic [D*32-1:0] add_a, add_b;

  genvar g;
  generate
    for (g = 0; g < D; g++) begin : g_lane
      // mulA: scalar x this lane's state element
      atlas_fp32_mul u_mA (
        .clk(clk), .rst_n(rst_n), .in_valid(mul_in),
        .a(scal_a), .b(rowd_q[g*32 +: 32]),
        .out_valid(mulA_vv[g]), .result(mulA_y[g*32 +: 32])
      );
      // mulB: scalar x this lane's u element (update pass only)
      atlas_fp32_mul u_mB (
        .clk(clk), .rst_n(rst_n), .in_valid(mul_in),
        .a(scal_b), .b(mulb_q[g*32 +: 32]),
        .out_valid(mulB_vv[g]), .result(mulB_y[g*32 +: 32])
      );
      atlas_fp32_add u_ad (
        .clk(clk), .rst_n(rst_n), .in_valid(add_in_mv | add_in_upd),
        .a(add_a_q[g*32 +: 32]), .b(add_b_q[g*32 +: 32]),
        .out_valid(add_vv[g]), .result(add_y[g*32 +: 32])
      );
    end
  endgenerate

  // Every lane is driven by the same valid and has identical latency, so
  // lane 0 speaks for all of them.  The other lanes' valids are left
  // unread rather than instantiating a spare unit to generate one.
  /* verilator lint_off UNUSEDSIGNAL */
  wire unused_lane_v = |mulA_vv[D-1:1] | |mulB_vv[D-1:1] | |add_vv[D-1:1];
  /* verilator lint_on UNUSEDSIGNAL */
  assign mulA_v = mulA_vv[0];
  assign mulB_v = mulB_vv[0];
  assign add_v  = add_vv[0];

  //-------------------------------------------------------------------------
  // Operand selection into the shared adder
  //-------------------------------------------------------------------------
  logic [D*32-1:0] part_rd;    // partial currently being accumulated into
  logic [D*32-1:0] part_0;     // partial 0, the fold destination
  logic [D*32-1:0] part_f;     // partial being folded in
  logic [D*32-1:0] neg_part_0; // -partial 0, for u = v - (S^T k)

  genvar gp;
  generate
    for (gp = 0; gp < D; gp++) begin : g_partrd
      assign part_rd[gp*32 +: 32] = part[(gp*PARTS + 32'(sel))*32 +: 32];
      assign part_0 [gp*32 +: 32] = part[(gp*PARTS)*32 +: 32];
      assign part_f [gp*32 +: 32] = part[(gp*PARTS + 32'(fold_i))*32 +: 32];
      assign neg_part_0[gp*32 +: 32] =
               {~part[(gp*PARTS)*32 + 31], part[(gp*PARTS)*32 +: 31]};
    end
  endgenerate

  always_comb begin
    unique case (st)
      // S <- diag(alpha) S + (beta k) u^T
      K_UPD, K_UPD_DRAIN: begin
        add_a = mulA_y;               // alpha[i] * S[i][j]
        add_b = mulB_y;               // (beta*k[i]) * u[j]
      end
      // fold the per-lane partials into partial 0
      K_FOLD, K_RD_FOLD: begin
        add_a = part_0;
        add_b = part_f;
      end
      // u = v - (S^T k); the subtraction is an add against the sign flip
      K_SUB: begin
        add_a = v_vec;
        add_b = neg_part_0;
      end
      // accumulating S^T k (or S^T q)
      default: begin
        add_a = part_rd;
        add_b = mulA_y;
      end
    endcase
  end

  integer i, p;

  assign busy     = (st != K_IDLE);
  assign st_rdata = S[32'(st_row)*D*32 +: D*32];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st     <= K_IDLE;
      done   <= 1'b0;
      S      <= '0;
      wb_wr  <= '0; wb_rd <= '0;
      rb_wr  <= '0; rb_rd <= '0;
      mul_in <= 1'b0; add_in_mv <= 1'b0; add_in_upd <= 1'b0;
      second_pass <= 1'b0;
      fold_busy <= 1'b0; sub_busy <= 1'b0;
    end else begin
      mul_in <= 1'b0; add_in_mv <= 1'b0; add_in_upd <= 1'b0;
      done   <= 1'b0;

      case (st)
        //---------------------------------------------------------------
        K_IDLE: begin
          if (st_we) S[32'(st_row)*D*32 +: D*32] <= st_wdata;
          if (start) begin
            for (i = 0; i < D; i = i + 1)
              for (p = 0; p < PARTS; p = p + 1)
                part[(i*PARTS + p)*32 +: 32] <= 32'h0;
            row         <= '0;
            n_done      <= '0;
            sel         <= '0;
            wb_wr       <= '0; wb_rd <= '0;
            second_pass <= 1'b0;
            st          <= K_MV;
          end
        end

        //---------------------------------------------------------------
        // u = S^T k   (and, on the second visit, o = S^T q)
        //---------------------------------------------------------------
        K_MV: begin
          if (row != IW'(D)) begin
            scal_a <= second_pass ? q_vec[32'(row)*32 +: 32] : k_vec[32'(row)*32 +: 32];
            rowd_q <= row_data;
            mul_in <= 1'b1;
            row    <= row + 1'b1;
          end else begin
            st <= second_pass ? K_RD_DRAIN : K_MV_DRAIN;
          end

          if (mulA_v) begin
            add_in_mv <= 1'b1;
            add_a_q   <= add_a;      // partial at the current sel
            add_b_q   <= add_b;      // this row's products
            wb[wb_wr[$clog2(WBD)-1:0]] <= sel;
            wb_wr     <= wb_wr + 1'b1;
            sel       <= (sel == PW'(PARTS-1)) ? '0 : (sel + 1'b1);
          end

          if (add_v) begin
            for (i = 0; i < D; i = i + 1)
              part[(i*PARTS + 32'(wb[wb_rd[$clog2(WBD)-1:0]]))*32 +: 32]
                <= add_y[i*32 +: 32];
            wb_rd  <= wb_rd + 1'b1;
            n_done <= n_done + 1'b1;
          end
        end

        K_MV_DRAIN, K_RD_DRAIN: begin
          if (mulA_v) begin
            add_in_mv <= 1'b1;
            add_a_q   <= add_a;      // partial at the current sel
            add_b_q   <= add_b;      // this row's products
            wb[wb_wr[$clog2(WBD)-1:0]] <= sel;
            wb_wr     <= wb_wr + 1'b1;
            sel       <= (sel == PW'(PARTS-1)) ? '0 : (sel + 1'b1);
          end
          if (add_v) begin
            for (i = 0; i < D; i = i + 1)
              part[(i*PARTS + 32'(wb[wb_rd[$clog2(WBD)-1:0]]))*32 +: 32]
                <= add_y[i*32 +: 32];
            wb_rd  <= wb_rd + 1'b1;
            n_done <= n_done + 1'b1;
          end
          if (n_done == IW'(D)) begin
            // Folding starts at partial 1: partial 0 is the destination.
            fold_i    <= (PW+1)'(1);
            fold_busy <= 1'b0;
            st        <= (st == K_MV_DRAIN) ? K_FOLD : K_RD_FOLD;
          end
        end

        //---------------------------------------------------------------
        // Fold the PARTS partials of each lane into partial 0.
        //---------------------------------------------------------------
        K_FOLD, K_RD_FOLD: begin
          if (!fold_busy) begin
            add_in_mv <= 1'b1; add_a_q <= add_a; add_b_q <= add_b;
            fold_busy <= 1'b1;
          end else if (add_v) begin
            for (i = 0; i < D; i = i + 1)
              part[(i*PARTS)*32 +: 32] <= add_y[i*32 +: 32];
            fold_busy <= 1'b0;
            if (fold_i == (PW+1)'(PARTS-1)) begin
              sub_busy <= 1'b0;
              st <= (st == K_FOLD) ? K_SUB : K_DONE;
            end else begin
              fold_i <= fold_i + 1'b1;
            end
          end
        end

        //---------------------------------------------------------------
        // u = v - (S^T k)
        //---------------------------------------------------------------
        K_SUB: begin
          if (!sub_busy) begin
            add_in_mv <= 1'b1; add_a_q <= add_a; add_b_q <= add_b;
            sub_busy  <= 1'b1;
          end else if (add_v) begin
            for (i = 0; i < D; i = i + 1)
              u_vec[i*32 +: 32] <= add_y[i*32 +: 32];
            bk_busy <= 1'b0;
            st      <= K_BK;
          end
        end

        //---------------------------------------------------------------
        // beta * k, all rows at once, using the lane multipliers.
        //---------------------------------------------------------------
        K_BK: begin
          if (!bk_busy) begin
            scal_b  <= beta;
            mulb_q  <= k_vec;
            mul_in  <= 1'b1;
            bk_busy <= 1'b1;
          end else if (mulB_v) begin
            bk_vec <= mulB_y;
            row    <= '0;
            n_done <= '0;
            rb_wr  <= '0; rb_rd <= '0;
            st     <= K_UPD;
          end
        end

        //---------------------------------------------------------------
        // S <- diag(alpha) S + (beta k) u^T
        //---------------------------------------------------------------
        K_UPD: begin
          if (row != IW'(D)) begin
            scal_a <= alpha_vec[32'(row)*32 +: 32];
            scal_b <= bk_vec[32'(row)*32 +: 32];
            rowd_q <= row_data;
            mulb_q <= u_vec;
            mul_in <= 1'b1;
            rb[rb_wr[$clog2(WBD)-1:0]] <= row;
            rb_wr  <= rb_wr + 1'b1;
            row    <= row + 1'b1;
          end else begin
            st <= K_UPD_DRAIN;
          end

          if (mulA_v) begin
            add_in_upd <= 1'b1;
            add_a_q    <= add_a;   // alpha[i]*S[i][j]
            add_b_q    <= add_b;   // (beta*k[i])*u[j]
          end

          if (add_v) begin
            S[32'(rb[rb_rd[$clog2(WBD)-1:0]])*D*32 +: D*32] <= add_y;
            rb_rd  <= rb_rd + 1'b1;
            n_done <= n_done + 1'b1;
          end
        end

        K_UPD_DRAIN: begin
          if (mulA_v) begin
            add_in_upd <= 1'b1;
            add_a_q    <= add_a;   // alpha[i]*S[i][j]
            add_b_q    <= add_b;   // (beta*k[i])*u[j]
          end
          if (add_v) begin
            S[32'(rb[rb_rd[$clog2(WBD)-1:0]])*D*32 +: D*32] <= add_y;
            rb_rd  <= rb_rd + 1'b1;
            n_done <= n_done + 1'b1;
          end
          if (n_done == IW'(D)) begin
            for (i = 0; i < D; i = i + 1)
              for (p = 0; p < PARTS; p = p + 1)
                part[(i*PARTS + p)*32 +: 32] <= 32'h0;
            row         <= '0;
            n_done      <= '0;
            sel         <= '0;
            wb_wr       <= '0; wb_rd <= '0;
            second_pass <= 1'b1;
            st          <= K_MV;
          end
        end

        //---------------------------------------------------------------
        K_DONE: begin
          for (i = 0; i < D; i = i + 1)
            o_vec[i*32 +: 32] <= part[(i*PARTS + 0)*32 +: 32];
          done <= 1'b1;
          st   <= K_IDLE;
        end

        default: st <= K_IDLE;
      endcase

    end
  end

endmodule
