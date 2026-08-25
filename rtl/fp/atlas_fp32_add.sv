//===========================================================================
// atlas_fp32_add -- IEEE-754 binary32 adder, round-to-nearest-even
//
// The dot-product engine reduces one MX block of 32 products per cycle.  A
// GEMM's K dimension is far longer than that, so block results are summed in
// binary32 here.  Accumulating in binary32 rather than continuing in fixed
// point is what keeps a long K from either overflowing or quietly losing the
// small blocks: each block carries its own E8M0 scale, so their magnitudes
// need not be related at all.
//
// Subnormal *inputs* are flushed to zero, and subnormal results are flushed
// to a signed zero.  At the point this sits -- accumulating products of FP4
// and FP8 values -- a binary32 denormal is some 2^-126 of the running total
// and cannot influence the final rounded result, so the alignment and
// normalisation hardware to support them would buy nothing.
//
// Two stages, one result per cycle.
//===========================================================================
`include "atlas_defs.svh"

module atlas_fp32_add (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        out_valid,
  output logic [31:0] result
);

  //-------------------------------------------------------------------------
  // S0 : unpack, order by magnitude, align, add or subtract
  //-------------------------------------------------------------------------
  logic        sa, sb;
  logic [7:0]  ea, eb;
  logic [22:0] ma, mb;

  assign {sa, ea, ma} = a;
  assign {sb, eb, mb} = b;

  wire a_zero = (ea == 8'd0);                    // subnormals flush to zero
  wire b_zero = (eb == 8'd0);
  wire a_inf  = (ea == 8'hFF) && (ma == 23'd0);
  wire a_nan  = (ea == 8'hFF) && (ma != 23'd0);
  wire b_inf  = (eb == 8'hFF) && (mb == 23'd0);
  wire b_nan  = (eb == 8'hFF) && (mb != 23'd0);

  // Order the operands so the larger magnitude is the alignment reference.
  wire a_ge = (ea > eb) || ((ea == eb) && (ma >= mb));

  wire        s_big  = a_ge ? sa : sb;
  wire        s_sml  = a_ge ? sb : sa;
  wire [7:0]  e_big  = a_ge ? ea : eb;
  wire [7:0]  e_sml  = a_ge ? eb : ea;
  wire [22:0] m_big  = a_ge ? ma : mb;
  wire [22:0] m_sml  = a_ge ? mb : ma;
  wire        z_big  = a_ge ? a_zero : b_zero;
  wire        z_sml  = a_ge ? b_zero : a_zero;

  // 24-bit significands, then three low bits for guard/round/sticky.
  wire [26:0] sig_big = z_big ? 27'd0 : {1'b1, m_big, 3'b000};
  wire [26:0] sig_sml = z_sml ? 27'd0 : {1'b1, m_sml, 3'b000};

  wire [7:0]  ediff   = e_big - e_sml;
  // Past 27 bits of shift the smaller operand can only ever be sticky.
  wire        far     = (ediff > 8'd27);
  wire [4:0]  shamt   = far ? 5'd27 : ediff[4:0];

  wire [26:0] sml_shf = sig_sml >> shamt;
  // Anything shifted out still has to influence the rounding decision.
  wire [26:0] lost_mask = ~({27{1'b1}} << shamt);
  wire        sticky_al = |(sig_sml & lost_mask);

  wire [26:0] sml_eff = {sml_shf[26:1], sml_shf[0] | sticky_al};

  // IEEE-754 sign of a zero result: two zeros of the same sign give that
  // zero, everything else (including exact cancellation of opposite nonzero
  // operands) gives +0 under round-to-nearest.
  wire        zero_sgn = z_big & z_sml & s_big & s_sml;

  wire subtract = s_big ^ s_sml;

  // Both the significand add/subtract and the rounding increment go through
  // the carry-select adder rather than being written as `+`.  yosys maps a
  // plain `+` to a ripple chain before ABC ever sees it, and ABC's rewriting
  // works on small windows, so it cannot rediscover a fast structure: the
  // 28-bit chain measured as this block's critical path, 66 majority gates
  // deep.  Subtraction is folded in as a + ~b + 1.
  wire [27:0] add_b = subtract ? ~{1'b0, sml_eff} : {1'b0, sml_eff};
  /* verilator lint_off UNUSEDSIGNAL */
  wire        raw_cout;
  /* verilator lint_on UNUSEDSIGNAL */
  wire [27:0] raw;

  atlas_cpa #(.W(28)) u_sig_add (
    .a({1'b0, sig_big}), .b(add_b), .cin(subtract), .sum(raw), .cout(raw_cout)
  );

  // Special-case resolution, decided here and carried to S1.
  //   NaN in, or Inf - Inf  -> quiet NaN
  //   Inf                   -> that Inf
  wire nan_out = a_nan | b_nan | (a_inf & b_inf & (sa ^ sb));
  wire inf_out = (a_inf | b_inf) & ~nan_out;
  wire inf_sgn = a_inf ? sa : sb;

  logic        p1_valid, p1_sgn, p1_nan, p1_inf, p1_infsgn, p1_zsgn;
  logic [27:0] p1_raw;
  logic [7:0]  p1_exp;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p1_valid <= 1'b0;
    else        p1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    p1_raw    <= raw;
    p1_exp    <= e_big;
    p1_sgn    <= s_big;
    p1_nan    <= nan_out;
    p1_inf    <= inf_out;
    p1_infsgn <= inf_sgn;
    p1_zsgn   <= zero_sgn;
  end

  //-------------------------------------------------------------------------
  // S1 : normalise, round to nearest even, pack
  //-------------------------------------------------------------------------
  logic [4:0] msb;
  logic       nonzero;

  atlas_msb_idx #(.W(28), .IW(5)) u_msb (
    .din(p1_raw), .idx(msb), .nonzero(nonzero)
  );

  // sig_big carries its implicit one at bit 26, so a result whose leading one
  // lands at bit m has unbiased exponent (m - 26) relative to e_big.
  logic [27:0]      shifted;
  logic [23:0]      signif;
  logic             rnd_bit, sticky, inc;
  wire  [24:0]      signif_r;
  logic             carry;
  // signif_f[23] is the implicit leading one and is intentionally dropped.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [23:0]      signif_f;
  /* verilator lint_on UNUSEDSIGNAL */
  logic signed [10:0] exp_adj, exp_fin;

  assign shifted  = p1_raw << (5'd27 - msb);
  assign signif   = shifted[27:4];
  assign rnd_bit  = shifted[3];
  assign sticky   = |shifted[2:0];
  assign inc      = rnd_bit & (sticky | signif[0]);
  /* verilator lint_off UNUSEDSIGNAL */
  wire signif_cout;
  /* verilator lint_on UNUSEDSIGNAL */
  atlas_cpa #(.W(25)) u_round (
    .a({1'b0, signif}), .b(25'd0), .cin(inc), .sum(signif_r), .cout(signif_cout)
  );
  assign carry    = signif_r[24];
  assign signif_f = carry ? signif_r[24:1] : signif_r[23:0];

  assign exp_adj  = $signed({3'b000, p1_exp}) + $signed(11'(msb)) - $signed(11'd26);
  assign exp_fin  = exp_adj + $signed({10'd0, carry});

  logic [31:0] s1_result;

  always_comb begin
    if (p1_nan)                           s1_result = 32'h7FC00000;
    else if (p1_inf)                      s1_result = {p1_infsgn, 8'hFF, 23'd0};
    else if (!nonzero)                    s1_result = {p1_zsgn, 31'd0};
    else if (exp_fin >= $signed(11'd255)) s1_result = {p1_sgn, 8'hFF, 23'd0};
    else if (exp_fin <= $signed(11'd0))   s1_result = {p1_sgn, 31'd0};
    else                                  s1_result = {p1_sgn, exp_fin[7:0], signif_f[22:0]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= p1_valid;
  end

  always_ff @(posedge clk) begin
    result <= s1_result;
  end

endmodule
