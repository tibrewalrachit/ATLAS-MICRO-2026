//===========================================================================
// atlas_rsqrt_f32 -- 1/sqrt(x) for binary32
//
// RMSNorm is the only thing on a Qwen layer that needs this, but it needs it
// once per token per norm, so it gets its own unit rather than being built out
// of a square root and a reciprocal.
//
// For x = m * 2^e with m in [1,2), the exponent has to be halved, so its
// parity decides the form:
//
//     e even :  1/sqrt(x) = (1/sqrt(m))  * 2^(-e/2)
//     e odd  :  1/sqrt(x) = (1/sqrt(2m)) * 2^(-(e-1)/2)
//
// Both significand functions land in (0.5, 1], so one table over [1,4) covers
// both cases -- an odd exponent is absorbed by doubling the significand, which
// costs nothing because it is just the parity bit prepended to the index.
//
// Negative inputs give NaN and zero gives Inf, following IEEE.  Accuracy is
// the table's, about 2^-19.
//
// Two stages, one result per cycle.
//===========================================================================
`include "atlas_defs.svh"

module atlas_rsqrt_f32 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] x,
  output logic        out_valid,
  output logic [31:0] y
);

  wire        s_x = x[31];
  wire [7:0]  e_x = x[30:23];
  wire [22:0] m_x = x[22:0];

  wire is_nan  = ((e_x == 8'hFF) && (m_x != 23'd0)) || (s_x && (e_x != 8'd0));
  wire is_inf  = (e_x == 8'hFF) && (m_x == 23'd0) && !s_x;
  wire is_zero = (e_x == 8'd0);

  // Unbiased exponent, and its parity.  Two's complement makes the low bit
  // the parity for negative exponents too, so no special case is needed.
  wire signed [9:0] e_unb  = $signed({2'b00, e_x}) - $signed(10'sd127);
  wire              parity = e_unb[0];

  logic [23:0] lut_base;
  logic [14:0] lut_delta;

  atlas_rsqrt_lut u_lut (
    .idx  ({parity, m_x[22:15]}),
    .base (lut_base),
    .delta(lut_delta)
  );

  // e even -> -e/2 - 1 ;  e odd -> -(e-1)/2 - 1.  Subtracting the parity
  // first makes the operand even in both cases, so the halving is an exact
  // arithmetic shift.
  wire signed [9:0] e_even = e_unb - $signed({9'd0, parity});
  wire signed [9:0] e_res  = -(e_even >>> 1) - $signed(10'sd1);

  logic        p1_valid, p1_nan, p1_inf, p1_zero;
  logic [14:0] p1_frac;
  logic [23:0] p1_base;
  logic [14:0] p1_delta;
  logic signed [9:0] p1_eres;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p1_valid <= 1'b0;
    else        p1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    p1_frac  <= m_x[14:0];
    p1_base  <= lut_base;
    p1_delta <= lut_delta;
    p1_eres  <= e_res;
    p1_nan   <= is_nan;
    p1_inf   <= is_inf;
    p1_zero  <= is_zero;
  end

  // The table's fraction is 16 bits and the position inside a segment is the
  // significand's low 15 bits, so this is {p1_frac, 1'b0} and the shift is
  // >> 15.  Operands widened first: a narrower context would size the multiply
  // too small and drop its top bits.
  /* verilator lint_off UNUSEDSIGNAL */
  wire [29:0] delta_scaled = 30'(p1_delta) * 30'(p1_frac);
  wire [23:0] rsq_m        = p1_base - 24'(delta_scaled >> 15);
  /* verilator lint_on UNUSEDSIGNAL */

  wire signed [10:0] biased = $signed({p1_eres[9], p1_eres}) + $signed(11'sd127);

  logic [31:0] y_c;
  always_comb begin
    if (p1_nan)                            y_c = 32'h7FC00000;   // x < 0 or NaN
    else if (p1_zero)                      y_c = 32'h7F800000;   // 1/sqrt(0)
    else if (p1_inf)                       y_c = 32'h00000000;   // 1/sqrt(inf)
    else if (biased >= $signed(11'sd255))  y_c = 32'h7F800000;
    else if (biased <= 0)                  y_c = 32'h00000000;
    else                                   y_c = {1'b0, biased[7:0], rsq_m[22:0]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= p1_valid;
  end

  always_ff @(posedge clk) begin
    y <= y_c;
  end

endmodule
