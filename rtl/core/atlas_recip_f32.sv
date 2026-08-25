//===========================================================================
// atlas_recip_f32 -- 1/x for binary32
//
// Used to normalise the MoE router's top-k weights, and by RMSNorm.
//
// A binary32 value is (-1)^s * m * 2^e with m in [1,2), so
//
//     1/x = (1/m) * 2^-e
//
// and the whole job is one table lookup on m plus an exponent negation.  The
// table returns 1/m in (0.5,1] scaled by 2^24, so the result is renormalised
// by borrowing one from the exponent -- no shifter and no leading-one detector.
//
// Accuracy is the table's, about 2^-19 relative.  Where a full binary32
// quotient is wanted a Newton-Raphson step would double the bits at the cost
// of two multipliers; nothing on this datapath needs it, since these
// reciprocals scale probabilities that are consumed at FP8 resolution.
//
// Two stages, one result per cycle.
//===========================================================================
`include "atlas_defs.svh"

module atlas_recip_f32 (
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

  wire is_nan  = (e_x == 8'hFF) && (m_x != 23'd0);
  wire is_inf  = (e_x == 8'hFF) && (m_x == 23'd0);
  wire is_zero = (e_x == 8'd0);                    // subnormals flush, 1/0 = Inf

  logic [23:0] lut_base;
  logic [15:0] lut_delta;

  atlas_recip_lut u_lut (
    .idx  (m_x[22:15]),
    .base (lut_base),
    .delta(lut_delta)
  );

  logic        p1_valid, p1_sign, p1_nan, p1_inf, p1_zero;
  logic [7:0]  p1_exp;
  logic [14:0] p1_frac;
  logic [23:0] p1_base;
  logic [15:0] p1_delta;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p1_valid <= 1'b0;
    else        p1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    p1_sign  <= s_x;
    p1_exp   <= e_x;
    p1_frac  <= m_x[14:0];
    p1_base  <= lut_base;
    p1_delta <= lut_delta;
    p1_nan   <= is_nan;
    p1_inf   <= is_inf;
    p1_zero  <= is_zero;
  end

  // The significand's low 15 bits give the position inside the segment.  The
  // table's fraction is 16 bits, so this is {p1_frac, 1'b0} and the shift
  // becomes >> 15.  Operands are widened before multiplying: sized by the
  // surrounding cast the product would lose its top bits.
  // recip_m[23] is the leading one of 1/m written as 1.f * 2^-1; it is
  // implicit in binary32 and dropped when packing.
  /* verilator lint_off UNUSEDSIGNAL */
  wire [30:0] delta_scaled = 31'(p1_delta) * 31'(p1_frac);
  wire [23:0] recip_m      = p1_base - 24'(delta_scaled >> 15);
  /* verilator lint_on UNUSEDSIGNAL */

  // 1/m lies in (0.5, 1], so the value is recip_m * 2^-24 and needs one
  // borrow from the exponent to be written as 1.f * 2^-1.
  //   1/x = (1/m) * 2^-(e-127)  ->  biased exponent 254 - e - 1
  wire signed [10:0] biased = $signed(11'sd253) - $signed({3'b000, p1_exp});

  logic [31:0] y_c;
  always_comb begin
    if (p1_nan)                            y_c = 32'h7FC00000;
    else if (p1_zero)                      y_c = {p1_sign, 8'hFF, 23'd0};  // 1/0
    else if (p1_inf)                       y_c = {p1_sign, 31'd0};         // 1/inf
    else if (biased >= $signed(11'sd255))  y_c = {p1_sign, 8'hFF, 23'd0};
    else if (biased <= 0)                  y_c = {p1_sign, 31'd0};
    else                                   y_c = {p1_sign, biased[7:0], recip_m[22:0]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= p1_valid;
  end

  always_ff @(posedge clk) begin
    y <= y_c;
  end

endmodule
