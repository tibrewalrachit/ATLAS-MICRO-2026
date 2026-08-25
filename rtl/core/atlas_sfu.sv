//===========================================================================
// atlas_sfu -- shared special-function unit
//
// Exponential, reciprocal, reciprocal square root and SiLU.  These are an
// order of magnitude larger than an add or a multiply -- each carries a
// 256-entry table -- so they are not replicated per lane.  A cloud ATLAS core
// has 480 vector lanes; giving every one of them its own table would dwarf the
// MAC array it is meant to support.
//
// The rates a layer actually needs justify the split: elementwise work is
// per-element on every element, while transcendentals appear once per
// activation (SiLU), once per row (RMSNorm's rsqrt) or once per score
// (softmax's exp).  atlas_vector_unit therefore instantiates one of these per
// SFU_RATIO lanes and lets the throughput difference show up as cycles.
//
// All four operations are padded to a common latency so results leave in the
// order they were issued whatever the mix.
//===========================================================================
`include "atlas_defs.svh"

module atlas_sfu (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [1:0]  op,
  input  logic [31:0] x,
  output logic        out_valid,
  output logic [31:0] y
);

  localparam logic [1:0] SOP_EXP   = 2'd0;
  localparam logic [1:0] SOP_RECIP = 2'd1;
  localparam logic [1:0] SOP_RSQRT = 2'd2;
  localparam logic [1:0] SOP_SILU  = 2'd3;

  localparam int unsigned EXP_LAT  = 3;
  localparam int unsigned RCP_LAT  = 2;
  localparam int unsigned RSQ_LAT  = 2;
  localparam int unsigned SILU_LAT = 9;
  localparam int unsigned LAT      = SILU_LAT;    // the longest

  logic        e_v, r_v, q_v, s_v;
  logic [31:0] e_y, r_y, q_y, s_y;

  atlas_exp_f32   u_exp (.clk(clk), .rst_n(rst_n),
                         .in_valid(in_valid && (op == SOP_EXP)),
                         .x(x), .out_valid(e_v), .y(e_y));
  atlas_recip_f32 u_rcp (.clk(clk), .rst_n(rst_n),
                         .in_valid(in_valid && (op == SOP_RECIP)),
                         .x(x), .out_valid(r_v), .y(r_y));
  atlas_rsqrt_f32 u_rsq (.clk(clk), .rst_n(rst_n),
                         .in_valid(in_valid && (op == SOP_RSQRT)),
                         .x(x), .out_valid(q_v), .y(q_y));
  atlas_silu_f32  u_slu (.clk(clk), .rst_n(rst_n),
                         .in_valid(in_valid && (op == SOP_SILU)),
                         .x(x), .out_valid(s_v), .y(s_y));

  /* verilator lint_off UNUSEDSIGNAL */
  wire unused_v = e_v | r_v | q_v | s_v;
  /* verilator lint_on UNUSEDSIGNAL */

  // Each result is held until the common latency expires, so a short operation
  // issued after a long one cannot overtake it.
  logic [31:0] hold_exp, hold_rcp, hold_rsq;

  always_ff @(posedge clk) begin
    if (e_v) hold_exp <= e_y;
    if (r_v) hold_rcp <= r_y;
    if (q_v) hold_rsq <= q_y;
  end

  logic [1:0] op_pipe  [0:LAT-1];
  logic       vld_pipe [0:LAT-1];

  integer d;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (d = 0; d < LAT; d = d + 1) vld_pipe[d] <= 1'b0;
    end else begin
      vld_pipe[0] <= in_valid;
      for (d = 1; d < LAT; d = d + 1) vld_pipe[d] <= vld_pipe[d-1];
    end
  end

  always_ff @(posedge clk) begin
    op_pipe[0] <= op;
    for (d = 1; d < LAT; d = d + 1) op_pipe[d] <= op_pipe[d-1];
  end

  always_comb begin
    unique case (op_pipe[LAT-1])
      SOP_EXP:   y = hold_exp;
      SOP_RECIP: y = hold_rcp;
      SOP_RSQRT: y = hold_rsq;
      default:   y = s_y;                        // SiLU already sets the pace
    endcase
  end

  assign out_valid = vld_pipe[LAT-1];

endmodule
