//===========================================================================
// atlas_norm_round_f32 -- normalise a fixed-point accumulator to IEEE binary32
//
// Takes a sign bit and an unsigned magnitude whose true value is
//
//     (-1)^sgn * mag * 2^exp_off
//
// and produces the correctly rounded binary32 encoding, round-to-nearest-even.
//
// The caller supplies the magnitude already formed.  Taking a signed value and
// negating it here would put a full-width carry chain in front of the
// leading-one detector, and that chain measured as the critical path of the
// whole dot-product engine.
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
  input  logic             sgn,
  input  logic [SUM_W-1:0] mag,
  input  logic [EXP_W-1:0] exp_off,
  input  logic             nan_in,
  output logic [31:0]      f32
);

  localparam int unsigned MSB_W = $clog2(SUM_W);

  // The normalisation window has to be at least as wide as the binary32
  // significand plus its round bit.  A narrow accumulator -- which is what
  // sizing the guard window to the format gives, e.g. 18 bits for an
  // MXFP4-only build -- would otherwise be sliced past its own top bit.
  // Widening here costs nothing: the extension is constant zero, so the shift
  // and the slices below collapse in synthesis.
  localparam int unsigned NW    = (SUM_W < 25) ? 25 : SUM_W;
  localparam int unsigned SH_W  = $clog2(NW) + 1;

  logic [MSB_W:0]   msb;      // index of the leading one
  logic             is_zero;

  // Leading-one position from a balanced tree.  A linear priority scan over
  // SUM_W bits would be the longest path in the block by a wide margin.
  logic [MSB_W-1:0] msb_i;
  logic             nonzero;

  atlas_msb_idx #(.W(SUM_W), .IW(MSB_W)) u_msb (
    .din(mag), .idx(msb_i), .nonzero(nonzero)
  );

  always_comb begin
    msb     = {1'b0, msb_i};
    is_zero = ~nonzero;
  end

  // Left-align so the leading one sits at bit SUM_W-1, then the top 24 bits
  // are the significand and the remainder feeds round/sticky.
  logic [NW-1:0]     norm;
  logic [23:0]       signif;
  logic              round_bit, sticky, inc;
  logic [24:0]       signif_r;

  logic [SH_W-1:0] shamt;
  assign shamt     = SH_W'(NW-1) - SH_W'(msb);
  assign norm      = NW'(mag) << shamt;
  assign signif    = norm[NW-1 -: 24];
  assign round_bit = norm[NW-25];
  // With NW == 25 there is nothing below the round bit, so the reduction has
  // no bits to take and sticky is constant zero.
  generate
    if (NW >= 26) begin : g_sticky
      assign sticky = |norm[NW-26:0];
    end else begin : g_no_sticky
      assign sticky = 1'b0;
    end
  endgenerate

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
