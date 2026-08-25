//===========================================================================
// atlas_norm_round_f32 -- normalise a fixed-point accumulator to IEEE binary32
//
// Takes a two's-complement integer `sum` whose true value is
//
//     sum * 2^exp_off
//
// and produces the correctly rounded binary32 encoding, round-to-nearest-even.
// Overflow saturates to Inf and underflow flushes to a signed zero, which is
// the behaviour an inference datapath wants (no subnormal FP32 output path is
// built, since a denormal result here is already far below FP4/FP8 resolution).
//
// Combinational; the caller registers the output.
//===========================================================================
`include "atlas_defs.svh"

module atlas_norm_round_f32 #(
  parameter int unsigned SUM_W = ATLAS_SUM_W,   // width of `sum`, signed
  parameter int unsigned EXP_W = 12             // width of `exp_off`, signed
) (
  input  logic [SUM_W-1:0] sum,
  input  logic [EXP_W-1:0] exp_off,
  input  logic             nan_in,
  output logic [31:0]      f32
);

  localparam int unsigned MSB_W = $clog2(SUM_W);

  logic             sgn;
  logic [SUM_W-1:0] mag;
  logic [MSB_W:0]   msb;      // index of the leading one
  logic             is_zero;

  // Magnitude.  |sum| always fits in SUM_W bits: the tree grows by exactly
  // LOG2N bits over an ALIGN_W+1-bit signed input, so -2^(SUM_W-1) is never
  // reachable and the negation cannot overflow.
  assign sgn = sum[SUM_W-1];
  assign mag = sgn ? (~sum + 1'b1) : sum;

  // Leading-one position.  Written as a descending priority scan; synthesis
  // maps this to a standard priority encoder.
  integer k;
  always_comb begin
    msb     = '0;
    is_zero = 1'b1;
    for (k = 0; k < SUM_W; k = k + 1) begin
      if (mag[k]) begin
        msb     = k[MSB_W:0];
        is_zero = 1'b0;
      end
    end
  end

  // Left-align so the leading one sits at bit SUM_W-1, then the top 24 bits
  // are the significand and the remainder feeds round/sticky.
  logic [SUM_W-1:0]  norm;
  logic [23:0]       signif;
  logic              round_bit, sticky, inc;
  logic [24:0]       signif_r;

  logic [MSB_W:0] shamt;
  assign shamt     = (MSB_W+1)'(SUM_W-1) - msb;
  assign norm      = mag << shamt;
  assign signif    = norm[SUM_W-1 -: 24];
  assign round_bit = norm[SUM_W-25];
  assign sticky    = |norm[SUM_W-26:0];
  assign inc       = round_bit & (sticky | signif[0]);
  assign signif_r  = {1'b0, signif} + {24'd0, inc};

  // Rounding may carry out of the 24-bit significand into 2.0.
  logic              carry;
  // signif_f[23] is the implicit leading one; it is intentionally dropped.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [23:0]       signif_f;
  /* verilator lint_on UNUSEDSIGNAL */
  assign carry    = signif_r[24];
  assign signif_f = carry ? signif_r[24:1] : signif_r[23:0];

  localparam int unsigned          EW  = EXP_W + 2;
  localparam logic signed [EW-1:0] F32_BIAS_C = EW'(127);
  localparam logic signed [EW-1:0] F32_EMAX_C = EW'(255);

  logic signed [EW-1:0] unbiased, biased;
  assign unbiased = $signed(EW'({{(EW-(MSB_W+1)){1'b0}}, msb}))
                  + $signed(EW'({{(EW-EXP_W){exp_off[EXP_W-1]}}, exp_off}))
                  + $signed(EW'({{(EW-1){1'b0}}, carry}));
  assign biased   = unbiased + F32_BIAS_C;

  always_comb begin
    if (nan_in)                          f32 = 32'h7FC00000;             // qNaN
    else if (is_zero)                    f32 = {sgn, 31'd0};             // +/-0
    else if (biased >= F32_EMAX_C)       f32 = {sgn, 8'hFF, 23'd0};      // +/-Inf
    else if (biased <= 0)                f32 = {sgn, 31'd0};             // flush
    else                                 f32 = {sgn, biased[7:0], signif_f[22:0]};
  end

endmodule
