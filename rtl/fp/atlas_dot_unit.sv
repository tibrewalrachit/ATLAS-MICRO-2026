//===========================================================================
// atlas_dot_unit -- MX block dot product for FP4/FP8 operands
//
// Computes  sum_{i=0}^{N-1} a[i]*b[i] * 2^(scale_a-127) * 2^(scale_b-127)
// and returns it as IEEE binary32.  N is the OCP microscaling block size (32),
// and scale_a/scale_b are the E8M0 shared block exponents.
//
// The datapath is integer, not floating point.  Each operand decodes to
// (sign, exp, mant4) with an *integer* significand, so a product is an 8-bit
// integer plus an exponent sum.  Products are aligned to the block's largest
// product exponent inside a GUARD_W-bit window and summed in two's complement.
// Consequences:
//
//   * For MXFP4 the entire reduction is bit-exact -- the product exponent
//     spread of an E2M1 block can never exceed the guard window.
//   * For FP8 the only error is the bounded alignment window; measured worst
//     case is 2^-22 of the largest product, versus the 2^-4 resolution of the
//     inputs themselves.
//   * No floating-point multiplier is instantiated anywhere in the array.
//
// Pipeline (4 stages, one result per cycle):
//   S0  decode operands, form products
//   S1  reduce to the block's maximum product exponent
//   S2  align products, two's-complement adder tree
//   S3  normalise and round to binary32
//===========================================================================
`include "atlas_defs.svh"

module atlas_dot_unit #(
  parameter int unsigned N       = ATLAS_DOT_N,
  parameter int unsigned LOG2N   = ATLAS_DOT_LOG2N,
  parameter int unsigned GUARD_W = ATLAS_GUARD_W
) (
  input  logic             clk,
  input  logic             rst_n,

  input  logic             in_valid,
  input  logic [1:0]       fmt_a,
  input  logic [1:0]       fmt_b,
  input  logic [N*8-1:0]   a_codes,   // FP4 uses the low nibble of each byte
  input  logic [N*8-1:0]   b_codes,
  input  logic [7:0]       scale_a,   // E8M0 block scale
  input  logic [7:0]       scale_b,

  output logic             out_valid,
  output logic [31:0]      out_f32
);

  localparam int unsigned PEXP_W  = ATLAS_PROD_EXP_W;              // 7, signed
  localparam int unsigned PMANT_W = ATLAS_PROD_MANT_W;             // 8
  localparam int unsigned ALIGN_W = PMANT_W + GUARD_W;             // 28
  localparam int unsigned TERM_W  = ALIGN_W + 1;                   // 29, signed
  localparam int unsigned SUM_W   = TERM_W + LOG2N;                // 34
  localparam int unsigned EXP_W   = 12;

  // Sentinel for lanes that contribute nothing, so they never win the max.
  localparam logic signed [PEXP_W-1:0] EXP_SENTINEL = -(2**(PEXP_W-1));

  //-------------------------------------------------------------------------
  // S0 : decode and multiply
  //-------------------------------------------------------------------------
  logic [N*PEXP_W-1:0]  s0_exp;
  logic [N*PMANT_W-1:0] s0_mant;
  logic [N-1:0]         s0_sign, s0_vld;
  logic [N-1:0]         s0_nan;

  genvar i;
  generate
    for (i = 0; i < N; i++) begin : g_lane
      logic [ATLAS_IFP_W-1:0] ifa, ifb;

      atlas_fp_decode u_da (.fmt(fmt_a), .code(a_codes[i*8 +: 8]), .ifp(ifa));
      atlas_fp_decode u_db (.fmt(fmt_b), .code(b_codes[i*8 +: 8]), .ifp(ifb));

      wire signed [ATLAS_IFP_EXP_W-1:0] ea = `ATLAS_IFP_EXP(ifa);
      wire signed [ATLAS_IFP_EXP_W-1:0] eb = `ATLAS_IFP_EXP(ifb);
      wire        [ATLAS_IFP_MANT_W-1:0] ma = `ATLAS_IFP_MANT(ifa);
      wire        [ATLAS_IFP_MANT_W-1:0] mb = `ATLAS_IFP_MANT(ifb);
      wire                               za = `ATLAS_IFP_ZERO(ifa);
      wire                               zb = `ATLAS_IFP_ZERO(ifb);

      wire vld = ~za & ~zb;
      wire signed [PEXP_W-1:0] esum =
             $signed({ea[ATLAS_IFP_EXP_W-1], ea}) + $signed({eb[ATLAS_IFP_EXP_W-1], eb});

      assign s0_sign[i]              = `ATLAS_IFP_SIGN(ifa) ^ `ATLAS_IFP_SIGN(ifb);
      assign s0_vld [i]              = vld;
      assign s0_nan [i]              = `ATLAS_IFP_NAN(ifa) | `ATLAS_IFP_NAN(ifb);
      // Force inactive lanes to the sentinel so the max tree ignores them.
      assign s0_exp [i*PEXP_W +: PEXP_W]   = vld ? esum : EXP_SENTINEL;
      assign s0_mant[i*PMANT_W +: PMANT_W] = ma * mb;
    end
  endgenerate

  logic                 p1_valid, p1_nan;
  logic [N*PEXP_W-1:0]  p1_exp;
  logic [N*PMANT_W-1:0] p1_mant;
  logic [N-1:0]         p1_sign, p1_vld;
  logic [7:0]           p1_sa, p1_sb;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p1_valid <= 1'b0;
    end else begin
      p1_valid <= in_valid;
      p1_exp   <= s0_exp;
      p1_mant  <= s0_mant;
      p1_sign  <= s0_sign;
      p1_vld   <= s0_vld;
      // An MX block scale of 0xFF is the E8M0 NaN encoding.
      p1_nan   <= (|s0_nan) | (scale_a == ATLAS_SCALE_NAN) | (scale_b == ATLAS_SCALE_NAN);
      p1_sa    <= scale_a;
      p1_sb    <= scale_b;
    end
  end

  //-------------------------------------------------------------------------
  // S1 : block maximum product exponent
  //-------------------------------------------------------------------------
  logic [PEXP_W-1:0] s1_maxexp;
  atlas_max_tree #(.N(N), .W(PEXP_W)) u_max (
    .din (p1_exp),
    .dmax(s1_maxexp)
  );

  logic                 p2_valid, p2_nan, p2_anyvld;
  logic [N*PEXP_W-1:0]  p2_exp;
  logic [N*PMANT_W-1:0] p2_mant;
  logic [N-1:0]         p2_sign, p2_vld;
  logic [PEXP_W-1:0]    p2_maxexp;
  logic [7:0]           p2_sa, p2_sb;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p2_valid <= 1'b0;
    end else begin
      p2_valid  <= p1_valid;
      p2_exp    <= p1_exp;
      p2_mant   <= p1_mant;
      p2_sign   <= p1_sign;
      p2_vld    <= p1_vld;
      p2_maxexp <= s1_maxexp;
      p2_nan    <= p1_nan;
      p2_anyvld <= |p1_vld;
      p2_sa     <= p1_sa;
      p2_sb     <= p1_sb;
    end
  end

  //-------------------------------------------------------------------------
  // S2 : align to the block maximum, then two's-complement adder tree
  //-------------------------------------------------------------------------
  logic [N*TERM_W-1:0] s2_term;

  generate
    for (i = 0; i < N; i++) begin : g_align
      wire signed [PEXP_W-1:0] e   = $signed(p2_exp[i*PEXP_W +: PEXP_W]);
      wire signed [PEXP_W:0]   dif = $signed({p2_maxexp[PEXP_W-1], p2_maxexp})
                                   - $signed({e[PEXP_W-1], e});
      // dif is non-negative by construction; anything past the guard window
      // contributes nothing and is flushed rather than wrapped.
      wire drop = dif[PEXP_W] | (dif >= $signed((PEXP_W+1)'(ALIGN_W)));

      wire [ALIGN_W-1:0] wide  = {p2_mant[i*PMANT_W +: PMANT_W], {GUARD_W{1'b0}}};
      wire [ALIGN_W-1:0] shf   = wide >> dif[PEXP_W-1:0];
      wire [ALIGN_W-1:0] mag   = (p2_vld[i] & ~drop) ? shf : {ALIGN_W{1'b0}};

      wire signed [TERM_W-1:0] pos = $signed({1'b0, mag});
      assign s2_term[i*TERM_W +: TERM_W] = p2_sign[i] ? (~pos + 1'b1) : pos;
    end
  endgenerate

  logic [SUM_W-1:0] s2_sum;
  atlas_add_tree #(.N(N), .W(TERM_W)) u_add (
    .din (s2_term),
    .dsum(s2_sum)
  );

  logic             p3_valid, p3_nan;
  logic [SUM_W-1:0] p3_sum;
  logic [EXP_W-1:0] p3_expoff;

  // value = sum * 2^(maxexp - GUARD_W) * 2^(scale_a-127) * 2^(scale_b-127)
  wire signed [EXP_W-1:0] s2_expoff =
        $signed(EXP_W'({{(EXP_W-PEXP_W){p2_maxexp[PEXP_W-1]}}, p2_maxexp}))
      - $signed(EXP_W'(GUARD_W))
      + $signed(EXP_W'({{(EXP_W-8){1'b0}}, p2_sa}))
      + $signed(EXP_W'({{(EXP_W-8){1'b0}}, p2_sb}))
      - $signed(EXP_W'(254));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p3_valid <= 1'b0;
    end else begin
      p3_valid  <= p2_valid;
      // A block with no active lane is an exact zero, not a scaled sum.
      p3_sum    <= p2_anyvld ? s2_sum : {SUM_W{1'b0}};
      p3_expoff <= s2_expoff;
      p3_nan    <= p2_nan;
    end
  end

  //-------------------------------------------------------------------------
  // S3 : normalise and round
  //-------------------------------------------------------------------------
  logic [31:0] s3_f32;
  atlas_norm_round_f32 #(.SUM_W(SUM_W), .EXP_W(EXP_W)) u_norm (
    .sum    (p3_sum),
    .exp_off(p3_expoff),
    .nan_in (p3_nan),
    .f32    (s3_f32)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_f32   <= 32'd0;
    end else begin
      out_valid <= p3_valid;
      out_f32   <= s3_f32;
    end
  end

endmodule
