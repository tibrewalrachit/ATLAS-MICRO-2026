//===========================================================================
// atlas_dot_unit -- MX block dot product for FP4/FP8 operands
//
// Computes  sum_{i=0}^{N-1} a[i]*b[i] * 2^(scale_a-127) * 2^(scale_b-127)
// and returns it as IEEE binary32.  N is the OCP microscaling block size (32)
// and scale_a/scale_b are the E8M0 shared block exponents.
//
// ---------------------------------------------------------------------------
// Why the datapath is integer
// ---------------------------------------------------------------------------
// Each operand decodes to (sign, exp, mant4) with an *integer* significand, so
// a product is an 8-bit integer multiply plus an exponent add -- there is no
// floating-point multiplier anywhere in the array.  Products are aligned to
// the block's largest product exponent inside a GUARD_W-bit window and reduced
// in two's complement.  Consequences:
//
//   * MXFP4 is bit-exact: an E2M1 block's product exponents can never spread
//     wider than the guard window, so nothing is ever dropped.
//   * FP8 error is bounded by the alignment window alone; measured worst case
//     is 2^-22 of the largest product, against inputs whose own resolution is
//     2^-4.
//
// GUARD_W is a design point, not a constant.  What it has to be is a property
// of the *formats being multiplied*: the exactness requirement is the sum of
// the two operand formats' exponent spreads.
//
//     E2M1 spread  2      E4M3 spread 14      E5M2 spread 29
//
// So E2M1 x E2M1 needs 4 and E4M3 x E4M3 needs 28.  Building for MXFP4 alone
// lets GUARD_W drop from 20 to 4, which takes 24% out of the engine and is
// what lets it close 1 GHz -- but such a build is then restricted to FP4 x FP4
// and will quietly truncate a mixed pair.  The check at the bottom of this
// module catches that in simulation.
//
// ---------------------------------------------------------------------------
// Pipeline (6 stages, one MX block per cycle)
// ---------------------------------------------------------------------------
//   S0  decode operands, form products
//   S1  partial maximum of the product exponents (N/GRP groups)
//   S2  final maximum, per-lane alignment shift and drop flag
//   S3  barrel-shift the products, carry-save reduce each half of the block
//   S4  merge the two carry-save pairs, resolve value and magnitude
//   S5  normalise and round to binary32
//
// The stage boundaries are where they are because of measured timing on
// ASAP7, not by guesswork.  Three things dominated in turn and each was fixed
// structurally, because ABC's restructuring undoes hand-built gate-level
// tricks such as an explicit fast adder:
//
//   * A balanced tree of ordinary adders put five carry propagations in
//     series (2.40 ns).  Carry-save reduction leaves exactly one.
//   * Normalising a signed accumulator put a full-width negation in front of
//     the leading-one detector.  S4 now forms the magnitude with two adders
//     running concurrently, so the negation costs a multiplexer.
//   * The exponent maximum is a tree of multi-bit comparators and was too
//     deep to fit beside the alignment logic, so it is split across S1/S2.
//===========================================================================
`include "atlas_defs.svh"

module atlas_dot_unit #(
  parameter int unsigned N       = ATLAS_DOT_N,
  parameter int unsigned LOG2N   = ATLAS_DOT_LOG2N,
  parameter int unsigned GUARD_W = ATLAS_GUARD_W,
  // Lanes per group in the first stage of the exponent maximum.  N/GRP partial
  // maxima cross the S1/S2 register, so this trades one stage's depth against
  // the other's without touching the datapath.
  parameter int unsigned GRP     = 8
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

  localparam int unsigned PEXP_W  = ATLAS_PROD_EXP_W;   // 7, signed
  localparam int unsigned PMANT_W = ATLAS_PROD_MANT_W;  // 8
  localparam int unsigned ALIGN_W = PMANT_W + GUARD_W;  // 28
  localparam int unsigned TERM_W  = ALIGN_W + 1;        // 29, signed
  localparam int unsigned SUM_W   = TERM_W + LOG2N;     // 34
  localparam int unsigned EXP_W   = 12;
  localparam int unsigned SH_W    = $clog2(ALIGN_W) + 1;
  localparam int unsigned NGRP    = N / GRP;

  // Lanes that contribute nothing are forced here so they can never win the
  // maximum, whatever their undefined exponent bits happen to be.
  localparam logic signed [PEXP_W-1:0] EXP_SENTINEL = -(2**(PEXP_W-1));

  genvar i, g;

  //=========================================================================
  // S0 : decode and multiply
  //=========================================================================
  logic [N*PEXP_W-1:0]  s0_exp;
  logic [N*PMANT_W-1:0] s0_mant;
  logic [N-1:0]         s0_sign, s0_vld, s0_nan;

  generate
    for (i = 0; i < N; i++) begin : g_lane
      logic [ATLAS_IFP_W-1:0] ifa, ifb;

      atlas_fp_decode u_da (.fmt(fmt_a), .code(a_codes[i*8 +: 8]), .ifp(ifa));
      atlas_fp_decode u_db (.fmt(fmt_b), .code(b_codes[i*8 +: 8]), .ifp(ifb));

      wire signed [ATLAS_IFP_EXP_W-1:0]  ea = `ATLAS_IFP_EXP(ifa);
      wire signed [ATLAS_IFP_EXP_W-1:0]  eb = `ATLAS_IFP_EXP(ifb);
      wire        [ATLAS_IFP_MANT_W-1:0] ma = `ATLAS_IFP_MANT(ifa);
      wire        [ATLAS_IFP_MANT_W-1:0] mb = `ATLAS_IFP_MANT(ifb);

      wire vld = ~`ATLAS_IFP_ZERO(ifa) & ~`ATLAS_IFP_ZERO(ifb);
      wire signed [PEXP_W-1:0] esum =
             $signed({ea[ATLAS_IFP_EXP_W-1], ea}) + $signed({eb[ATLAS_IFP_EXP_W-1], eb});

      assign s0_sign[i]                    = `ATLAS_IFP_SIGN(ifa) ^ `ATLAS_IFP_SIGN(ifb);
      assign s0_vld [i]                    = vld;
      assign s0_nan [i]                    = `ATLAS_IFP_NAN(ifa) | `ATLAS_IFP_NAN(ifb);
      assign s0_exp [i*PEXP_W  +: PEXP_W]  = vld ? esum : EXP_SENTINEL;
      assign s0_mant[i*PMANT_W +: PMANT_W] = ma * mb;
    end
  endgenerate

  // Only the valid bit carries reset.  Gating the ~700 datapath flops with
  // rst_n would turn the reset net into a block-wide enable: an enable flop
  // per bit, and the reset tree as the worst slew path in the design.  Their
  // contents are meaningless until the valid bit says otherwise.
  logic                 p1_valid, p1_nan;
  logic [N*PEXP_W-1:0]  p1_exp;
  logic [N*PMANT_W-1:0] p1_mant;
  logic [N-1:0]         p1_sign, p1_vld;
  logic [7:0]           p1_sa, p1_sb;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p1_valid <= 1'b0;
    else        p1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    p1_exp  <= s0_exp;
    p1_mant <= s0_mant;
    p1_sign <= s0_sign;
    p1_vld  <= s0_vld;
    // 0xFF is the E8M0 NaN encoding for a block scale.
    p1_nan  <= (|s0_nan) | (scale_a == ATLAS_SCALE_NAN) | (scale_b == ATLAS_SCALE_NAN);
    p1_sa   <= scale_a;
    p1_sb   <= scale_b;
  end

  //=========================================================================
  // S1 : partial maximum over groups of GRP lanes
  //=========================================================================
  logic [NGRP*PEXP_W-1:0] s1_pmax;

  generate
    for (g = 0; g < NGRP; g++) begin : g_pmax
      atlas_max_tree #(.N(GRP), .W(PEXP_W)) u_pmax (
        .din (p1_exp[g*GRP*PEXP_W +: GRP*PEXP_W]),
        .dmax(s1_pmax[g*PEXP_W +: PEXP_W])
      );
    end
  endgenerate

  logic                   p2_valid, p2_nan;
  logic [NGRP*PEXP_W-1:0] p2_pmax;
  logic [N*PEXP_W-1:0]    p2_exp;
  logic [N*PMANT_W-1:0]   p2_mant;
  logic [N-1:0]           p2_sign, p2_vld;
  logic [7:0]             p2_sa, p2_sb;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p2_valid <= 1'b0;
    else        p2_valid <= p1_valid;
  end

  always_ff @(posedge clk) begin
    p2_pmax <= s1_pmax;
    p2_exp  <= p1_exp;
    p2_mant <= p1_mant;
    p2_sign <= p1_sign;
    p2_vld  <= p1_vld;
    p2_nan  <= p1_nan;
    p2_sa   <= p1_sa;
    p2_sb   <= p1_sb;
  end

  //=========================================================================
  // S2 : final maximum, then the per-lane alignment shift
  //=========================================================================
  logic [PEXP_W-1:0] s2_maxexp;
  atlas_max_tree #(.N(NGRP), .W(PEXP_W)) u_fmax (
    .din (p2_pmax),
    .dmax(s2_maxexp)
  );

  logic [N*SH_W-1:0] s2_sh;
  logic [N-1:0]      s2_drop;

  generate
    for (i = 0; i < N; i++) begin : g_shamt
      wire signed [PEXP_W-1:0] e   = $signed(p2_exp[i*PEXP_W +: PEXP_W]);
      wire signed [PEXP_W:0]   dif = $signed({s2_maxexp[PEXP_W-1], s2_maxexp})
                                   - $signed({e[PEXP_W-1], e});
      // dif is non-negative by construction.  Anything at or beyond the guard
      // window contributes nothing and is flushed rather than wrapped.
      assign s2_drop[i]            = dif[PEXP_W] | (dif >= $signed((PEXP_W+1)'(ALIGN_W)));
      assign s2_sh[i*SH_W +: SH_W] = dif[SH_W-1:0];
    end
  endgenerate

  // value = sum * 2^(maxexp - GUARD_W) * 2^(scale_a-127) * 2^(scale_b-127)
  wire signed [EXP_W-1:0] s2_expoff =
        $signed(EXP_W'({{(EXP_W-PEXP_W){s2_maxexp[PEXP_W-1]}}, s2_maxexp}))
      - $signed(EXP_W'(GUARD_W))
      + $signed(EXP_W'({{(EXP_W-8){1'b0}}, p2_sa}))
      + $signed(EXP_W'({{(EXP_W-8){1'b0}}, p2_sb}))
      - $signed(EXP_W'(254));

  logic                 p3_valid, p3_nan, p3_anyvld;
  logic [N*SH_W-1:0]    p3_sh;
  logic [N-1:0]         p3_drop, p3_sign, p3_vld;
  logic [N*PMANT_W-1:0] p3_mant;
  logic [EXP_W-1:0]     p3_expoff;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p3_valid <= 1'b0;
    else        p3_valid <= p2_valid;
  end

  always_ff @(posedge clk) begin
    p3_sh     <= s2_sh;
    p3_drop   <= s2_drop;
    p3_mant   <= p2_mant;
    p3_sign   <= p2_sign;
    p3_vld    <= p2_vld;
    p3_anyvld <= |p2_vld;
    p3_expoff <= s2_expoff;
    p3_nan    <= p2_nan;
  end

  //=========================================================================
  // S3 : align, then carry-save reduce each half of the block
  //=========================================================================
  logic [N*SUM_W-1:0] s3_term;

  generate
    for (i = 0; i < N; i++) begin : g_align
      wire [ALIGN_W-1:0] wide = {p3_mant[i*PMANT_W +: PMANT_W], {GUARD_W{1'b0}}};
      wire [ALIGN_W-1:0] shf  = wide >> p3_sh[i*SH_W +: SH_W];
      wire [ALIGN_W-1:0] mag  = (p3_vld[i] & ~p3_drop[i]) ? shf : {ALIGN_W{1'b0}};

      wire signed [TERM_W-1:0] pos = $signed({1'b0, mag});
      wire signed [TERM_W-1:0] trm = p3_sign[i] ? (~pos + 1'b1) : pos;
      // The carry-save tree works modulo 2^SUM_W, so terms enter it already
      // sign-extended to the full accumulator width.
      assign s3_term[i*SUM_W +: SUM_W] = {{(SUM_W-TERM_W){trm[TERM_W-1]}}, trm};
    end
  endgenerate

  // Reducing each half separately means only four SUM_W vectors cross into S4,
  // instead of the N*SUM_W it would take to register the aligned terms.
  logic [SUM_W-1:0] s3_slo, s3_clo, s3_shi, s3_chi;

  atlas_csa_tree #(.N(N/2), .W(SUM_W)) u_csa_lo (
    .din  (s3_term[0             +: (N/2)*SUM_W]),
    .s_out(s3_slo), .c_out(s3_clo)
  );
  atlas_csa_tree #(.N(N/2), .W(SUM_W)) u_csa_hi (
    .din  (s3_term[(N/2)*SUM_W   +: (N/2)*SUM_W]),
    .s_out(s3_shi), .c_out(s3_chi)
  );

  logic             p4_valid, p4_nan;
  logic [SUM_W-1:0] p4_slo, p4_clo, p4_shi, p4_chi;
  logic [EXP_W-1:0] p4_expoff;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p4_valid <= 1'b0;
    else        p4_valid <= p3_valid;
  end

  always_ff @(posedge clk) begin
    // A block with no active lane is an exact zero, not a scaled sum.
    p4_slo    <= p3_anyvld ? s3_slo : {SUM_W{1'b0}};
    p4_clo    <= p3_anyvld ? s3_clo : {SUM_W{1'b0}};
    p4_shi    <= p3_anyvld ? s3_shi : {SUM_W{1'b0}};
    p4_chi    <= p3_anyvld ? s3_chi : {SUM_W{1'b0}};
    p4_expoff <= p3_expoff;
    p4_nan    <= p3_nan;
  end

  //=========================================================================
  // S4 : merge the pairs, resolve value and magnitude together
  //=========================================================================
  logic [4*SUM_W-1:0] s4_pack;
  logic [SUM_W-1:0]   s4_s, s4_c;

  assign s4_pack = {p4_chi, p4_shi, p4_clo, p4_slo};

  atlas_csa_tree #(.N(4), .W(SUM_W)) u_csa_merge (
    .din(s4_pack), .s_out(s4_s), .c_out(s4_c)
  );

  // The value and its negation are resolved by two adders running
  // concurrently, and the sign then selects between them.  Negating after the
  // add would chain a second full-width carry propagation behind the first.
  //   -(s + c) == (~s) + (~c) + 2
  // The +2 is absorbed into one carry-save level rather than an incrementer,
  // so both paths are exactly one carry propagation deep.
  localparam logic [SUM_W-1:0] TWO = SUM_W'(2);

  // ncry_pre's top bit is shifted out below -- the same modulo-2^SUM_W
  // behaviour the carry-save tree relies on, not a lost term.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [SUM_W-1:0] ns, nc, ncry_pre, nsum_v, ncry_v;
  /* verilator lint_on UNUSEDSIGNAL */
  assign ns       = ~s4_s;
  assign nc       = ~s4_c;
  assign nsum_v   = ns ^ nc ^ TWO;
  assign ncry_pre = (ns & nc) | (nc & TWO) | (ns & TWO);
  assign ncry_v   = {ncry_pre[SUM_W-2:0], 1'b0};

  // The accumulator is sized so neither sum can overflow, so these carry-outs
  // are never meaningful and are deliberately left unconnected.
  /* verilator lint_off UNUSEDSIGNAL */
  logic s4_cout_p, s4_cout_n;
  /* verilator lint_on UNUSEDSIGNAL */

  logic [SUM_W-1:0] s4_pos, s4_neg, s4_mag;
  logic             s4_sgn;

  atlas_cpa #(.W(SUM_W)) u_cpa_pos (
    .a(s4_s), .b(s4_c), .cin(1'b0), .sum(s4_pos), .cout(s4_cout_p));
  atlas_cpa #(.W(SUM_W)) u_cpa_neg (
    .a(nsum_v), .b(ncry_v), .cin(1'b0), .sum(s4_neg), .cout(s4_cout_n));

  assign s4_sgn = s4_pos[SUM_W-1];
  assign s4_mag = s4_sgn ? s4_neg : s4_pos;

  logic             p5_valid, p5_nan, p5_sgn;
  logic [SUM_W-1:0] p5_mag;
  logic [EXP_W-1:0] p5_expoff;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p5_valid <= 1'b0;
    else        p5_valid <= p4_valid;
  end

  always_ff @(posedge clk) begin
    p5_mag    <= s4_mag;
    p5_sgn    <= s4_sgn;
    p5_expoff <= p4_expoff;
    p5_nan    <= p4_nan;
  end

  //=========================================================================
  // S5 : normalise and round
  //=========================================================================
  logic [31:0] s5_f32;

  atlas_norm_round_f32 #(.SUM_W(SUM_W), .EXP_W(EXP_W)) u_norm (
    .sgn    (p5_sgn),
    .mag    (p5_mag),
    .exp_off(p5_expoff),
    .nan_in (p5_nan),
    .f32    (s5_f32)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= p5_valid;
  end

  always_ff @(posedge clk) begin
    out_f32 <= s5_f32;
  end


  //-------------------------------------------------------------------------
  // Format/guard-window check (simulation only).
  //
  // A narrowed build is exact only for the format pair it was sized for.
  // Feeding it a wider pair loses low-order products silently -- the result
  // stays plausible, just wrong -- so it is worth failing loudly instead.
  //-------------------------------------------------------------------------
`ifndef SYNTHESIS
  function automatic int fmt_spread(input logic [1:0] f);
    case (f)
      ATLAS_FMT_E2M1: fmt_spread = 2;
      ATLAS_FMT_E4M3: fmt_spread = 14;
      ATLAS_FMT_E5M2: fmt_spread = 29;
      default:        fmt_spread = 0;    // BF16 never reaches this array
    endcase
  endfunction

  logic warned;
  initial warned = 1'b0;

  always_ff @(posedge clk) begin
    if (in_valid && !warned) begin
      if (int'(GUARD_W) < (fmt_spread(fmt_a) + fmt_spread(fmt_b))) begin
        $display("%0t WARNING atlas_dot_unit: GUARD_W=%0d is too narrow for this format pair (needs %0d).",
                 $time, GUARD_W, fmt_spread(fmt_a) + fmt_spread(fmt_b));
        $display("         Products below the window are dropped; the result will be inexact.");
        warned <= 1'b1;
      end
    end
  end
`endif

endmodule
