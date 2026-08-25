//===========================================================================
// atlas_exp_f32 -- e^x for binary32, by way of 2^(x*log2 e)
//
// Needed in two places on a Qwen inference pass: the attention softmax, and
// the MoE router's expert weights.
//
// Range reduction is done in fixed point rather than floating point.  x is
// converted once to Q16.16, scaled by log2(e) there, and split into an integer
// part -- which becomes the result's exponent directly, no normalisation
// needed -- and a fraction that indexes rtl/gen/atlas_exp2_lut.sv.  Because
// 2^f lies in [1,2), the table's output *is* the significand: there is no
// leading-one detector and no shifter anywhere in this module.
//
// Accuracy is the table's, about 2^-19 relative (the generator reports it),
// against a binary32 significand of 2^-24.  Softmax weights and router
// probabilities are consumed at FP8-or-coarser resolution, so the last few
// bits buy nothing; spending a Newton iteration to recover them would cost a
// multiplier per lane.
//
// Three stages, one result per cycle.
//===========================================================================
`include "atlas_defs.svh"

module atlas_exp_f32 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] x,
  output logic        out_valid,
  output logic [31:0] y
);

  // log2(e) in Q2.30.
  //
  // The width of this constant matters more than it looks.  Its relative error
  // is amplified by |x| when forming x*log2(e), and that product is an
  // *exponent*, so an absolute error d there becomes a relative error of
  // ln2*d in the result.  A Q1.16 constant (error 4e-6) gives 2.4e-4 at
  // x = 88 -- two decimal digits worse than the table it feeds.  At Q2.30 the
  // constant contributes about 3e-8.
  localparam logic [30:0] LOG2E_Q30 = 31'd1549082005;

  //-------------------------------------------------------------------------
  // S0 : unpack and convert to Q16.16 fixed point
  //-------------------------------------------------------------------------
  wire        s_x = x[31];
  wire [7:0]  e_x = x[30:23];
  wire [22:0] m_x = x[22:0];

  wire is_nan  = (e_x == 8'hFF) && (m_x != 23'd0);
  wire is_inf  = (e_x == 8'hFF) && (m_x == 23'd0);
  wire is_zero = (e_x == 8'd0);                    // subnormals: e^x == 1

  wire signed [9:0] unb = $signed({2'b00, e_x}) - $signed(10'sd127);
  wire [23:0] sig = {1'b1, m_x};

  // x is carried as Q11.20.  Twenty fraction bits are not decoration: the
  // quantisation of x lands in the exponent, so with only 16 the result would
  // be limited to about 7.6e-6 relative -- worse than the table's 1.9e-6.  At
  // 20 bits it contributes 4.8e-7 and the table dominates again.
  //   value * 2^20 == sig * 2^(unb - 3)
  wire signed [9:0] sh = unb - $signed(10'sd3);

  logic [31:0] mag_fixed;
  logic        sat;

  always_comb begin
    sat = 1'b0;
    if (sh >= $signed(10'sd8)) begin
      // |x| >= 1024 here, far outside the range where e^x is finite in
      // binary32, so clamping costs nothing.
      mag_fixed = 32'h7FFF_FFFF;
      sat       = 1'b1;
    end else if (sh >= 0) begin
      mag_fixed = {8'd0, sig} << sh[3:0];
    end else if (sh > -$signed(10'sd24)) begin
      mag_fixed = {8'd0, sig} >> ((-sh) & 10'sd31);
    end else begin
      mag_fixed = 32'd0;                            // |x| far below 2^-20
    end
  end

  // The magnitude is carried through the multiply unsigned and the sign
  // applied afterwards.  Negating before a wide multiply would put a
  // full-width carry chain in front of it for nothing.
  logic        p1_valid, p1_nan, p1_inf, p1_zero, p1_sign, p1_sat;
  logic [31:0] p1_mag;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p1_valid <= 1'b0;
    else        p1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    p1_mag   <= mag_fixed;
    p1_nan   <= is_nan;
    p1_inf   <= is_inf;
    p1_zero  <= is_zero;
    p1_sign  <= s_x;
    p1_sat   <= sat;
  end

  //-------------------------------------------------------------------------
  // S1 : scale by log2(e), split integer and fraction, read the table
  //-------------------------------------------------------------------------
  // Q11.20 * Q2.30 -> Q13.50; keep Q13.20.  The discarded low bits sit far
  // below the table's resolution.
  //
  // Structural, not `*`: yosys expands a multiply into a ripple-carry array,
  // and this one measured 105 majority gates deep -- the critical path of this
  // block and, through the SFU, of everything that contains it.  log2(e) is a
  // constant with 16 set bits, so half the partial-product rows fold away
  // before the carry-save tree ever sees them.
  /* verilator lint_off UNUSEDSIGNAL */
  wire [63:0] t_full;
  /* verilator lint_on UNUSEDSIGNAL */

  atlas_mul_csa #(.W(32)) u_scale (
    .a(p1_mag), .b({1'b0, LOG2E_Q30}), .p(t_full)
  );

  // |x| is clamped below 1024, so the product cannot reach bit 63.
  wire [32:0] t_mag = t_full[62:30];

  // Apply the sign.  Two's complement negation gives floor semantics for the
  // integer part, which is what the exponent split below needs.
  /* verilator lint_off UNUSEDSIGNAL */
  wire t_cout;
  /* verilator lint_on UNUSEDSIGNAL */
  wire signed [32:0] t_q20;

  atlas_cpa #(.W(33)) u_sign (
    .a(p1_sign ? ~t_mag : t_mag), .b(33'd0), .cin(p1_sign),
    .sum(t_q20), .cout(t_cout)
  );

  /* verilator lint_off UNUSEDSIGNAL */
  wire signed [12:0] t_int  = t_q20[32:20];
  wire        [19:0] t_frac = t_q20[19:0];
  /* verilator lint_on UNUSEDSIGNAL */

  logic [23:0] lut_base;
  logic [15:0] lut_delta;

  atlas_exp2_lut u_lut (
    .idx  (t_frac[19:12]),
    .base (lut_base),
    .delta(lut_delta)
  );

  logic               p2_valid, p2_nan, p2_inf, p2_zero, p2_sign, p2_sat;
  logic signed [12:0] p2_int;
  logic [11:0]        p2_fl;
  logic [23:0]        p2_base;
  logic [15:0]        p2_delta;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p2_valid <= 1'b0;
    else        p2_valid <= p1_valid;
  end

  always_ff @(posedge clk) begin
    p2_int   <= t_int;
    p2_fl    <= t_frac[11:0];
    p2_base  <= lut_base;
    p2_delta <= lut_delta;
    p2_nan   <= p1_nan;
    p2_inf   <= p1_inf;
    p2_zero  <= p1_zero;
    p2_sign  <= p1_sign;
    p2_sat   <= p1_sat;
  end

  //-------------------------------------------------------------------------
  // S2 : interpolate and pack
  //-------------------------------------------------------------------------
  // p2_fl is the position inside the segment.  All twelve remaining fraction
  // bits are used: with only eight, f would be quantised to 2^-16 of the unit
  // and the result limited to ln2*2^-16 = 1.1e-5 relative -- five times worse
  // than the table it is interpolating.  The table's fraction is 16 bits, so
  // a 12-bit position means {p2_fl, 4'd0} and the shift collapses to >> 12.
  // interp[23] is the leading one of 2^f in [1,2) -- implicit in binary32 and
  // therefore intentionally dropped when packing below.
  /* verilator lint_off UNUSEDSIGNAL */
  // The product needs 16+12 bits.  Written inside a 24-bit cast the context
  // would size the multiply at 24 and silently drop its top bits, so the
  // operands are widened first.
  wire [27:0] delta_scaled = 28'(p2_delta) * 28'(p2_fl);
  wire [23:0] interp       = p2_base + 24'(delta_scaled >> 12);
  /* verilator lint_on UNUSEDSIGNAL */

  wire signed [17:0] biased = $signed({{5{p2_int[12]}}, p2_int}) + $signed(18'sd127);

  logic [31:0] y_c;
  always_comb begin
    if (p2_nan)                       y_c = 32'h7FC00000;
    else if (p2_inf)                  y_c = p2_sign ? 32'h00000000   // e^-inf
                                                    : 32'h7F800000;  // e^+inf
    else if (p2_zero)                 y_c = 32'h3F800000;            // e^0 == 1
    else if (p2_sat)                  y_c = p2_sign ? 32'h00000000
                                                    : 32'h7F800000;
    else if (biased >= $signed(18'sd255)) y_c = 32'h7F800000;
    else if (biased <= 0)             y_c = 32'h00000000;            // flush
    else                              y_c = {1'b0, biased[7:0], interp[22:0]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= p2_valid;
  end

  always_ff @(posedge clk) begin
    y <= y_c;
  end

endmodule
