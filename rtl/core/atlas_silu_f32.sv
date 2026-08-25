//===========================================================================
// atlas_silu_f32 -- SiLU (swish) activation for binary32
//
//     silu(x) = x * sigmoid(x) = x / (1 + e^-x)
//
// Qwen3 uses SiLU as its MLP gate (hidden_act "silu" in the model config), so
// every expert's up-projection passes through this.
//
// Built by composing the units that already exist rather than tabulating SiLU
// directly.  A direct table would need to cover a range where the function
// grows without bound, which means either a wide table or poor relative
// accuracy in the tail; going through e^-x keeps the tabulated part inside
// [1,2) where a 256-entry table is already at 2^-19.
//
// The chain is exp -> add 1 -> reciprocal -> multiply.  x itself is delayed to
// meet the multiplier, so the whole thing streams at one value per cycle.
//===========================================================================
`include "atlas_defs.svh"

module atlas_silu_f32 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] x,
  output logic        out_valid,
  output logic [31:0] y
);

  localparam int unsigned EXP_LAT = 3;
  localparam int unsigned ADD_LAT = 2;
  localparam int unsigned RCP_LAT = 2;
  localparam int unsigned PRE_LAT = EXP_LAT + ADD_LAT + RCP_LAT;   // before the multiply

  // e^-x
  logic        e_v;
  logic [31:0] e_y;
  atlas_exp_f32 u_exp (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
    .x({~x[31], x[30:0]}), .out_valid(e_v), .y(e_y)
  );

  // 1 + e^-x
  logic        a_v;
  logic [31:0] a_y;
  atlas_fp32_add u_add (
    .clk(clk), .rst_n(rst_n), .in_valid(e_v),
    .a(32'h3F800000), .b(e_y), .out_valid(a_v), .result(a_y)
  );

  // sigmoid(x) = 1 / (1 + e^-x)
  logic        r_v;
  logic [31:0] r_y;
  atlas_recip_f32 u_rcp (
    .clk(clk), .rst_n(rst_n), .in_valid(a_v),
    .x(a_y), .out_valid(r_v), .y(r_y)
  );

  // x, delayed to meet its sigmoid at the multiplier.
  logic [31:0] x_pipe [0:PRE_LAT-1];
  integer d;
  always_ff @(posedge clk) begin
    x_pipe[0] <= x;
    for (d = 1; d < PRE_LAT; d = d + 1) x_pipe[d] <= x_pipe[d-1];
  end

  atlas_fp32_mul u_mul (
    .clk(clk), .rst_n(rst_n), .in_valid(r_v),
    .a(x_pipe[PRE_LAT-1]), .b(r_y),
    .out_valid(out_valid), .result(y)
  );

endmodule
