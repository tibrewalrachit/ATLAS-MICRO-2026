//===========================================================================
// atlas_fp32_mul -- IEEE-754 binary32 multiplier, round-to-nearest-even
//
// Needed wherever a scalar scales a vector: MoE weight normalisation, the
// RMSNorm reciprocal applied to a row, softmax weights against values.
//
// Multiplication needs no alignment and no leading-zero detector -- the
// product of two significands in [1,2) is in [1,4), so at most a one-bit
// normalisation.  That makes this markedly cheaper than the adder.
//
// Subnormal inputs flush to zero and subnormal results flush to a signed
// zero, matching atlas_fp32_add.
//
// Two stages, one result per cycle.
//===========================================================================
`include "atlas_defs.svh"

module atlas_fp32_mul (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        out_valid,
  output logic [31:0] result
);

  wire        sa = a[31], sb = b[31];
  wire [7:0]  ea = a[30:23], eb = b[30:23];
  wire [22:0] ma = a[22:0],  mb = b[22:0];

  wire a_zero = (ea == 8'd0);
  wire b_zero = (eb == 8'd0);
  wire a_inf  = (ea == 8'hFF) && (ma == 23'd0);
  wire b_inf  = (eb == 8'hFF) && (mb == 23'd0);
  wire a_nan  = (ea == 8'hFF) && (ma != 23'd0);
  wire b_nan  = (eb == 8'hFF) && (mb != 23'd0);

  // 0 * Inf is the only new NaN a multiply can create.
  wire nan_out  = a_nan | b_nan | (a_zero & b_inf) | (b_zero & a_inf);
  wire inf_out  = (a_inf | b_inf) & ~nan_out;
  wire zero_out = (a_zero | b_zero) & ~nan_out & ~inf_out;

  // Structural multiplier, not `*`: see atlas_mul_csa for why.
  wire [47:0] prod;
  atlas_mul_csa #(.W(24)) u_mul (
    .a({1'b1, ma}), .b({1'b1, mb}), .p(prod)
  );
  wire signed [10:0] esum = $signed({3'b000, ea}) + $signed({3'b000, eb})
                          - $signed(11'sd127);

  logic        p1_valid, p1_sign, p1_nan, p1_inf, p1_zero;
  logic [47:0] p1_prod;
  logic signed [10:0] p1_exp;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) p1_valid <= 1'b0;
    else        p1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    p1_prod <= prod;
    p1_exp  <= esum;
    p1_sign <= sa ^ sb;
    p1_nan  <= nan_out;
    p1_inf  <= inf_out;
    p1_zero <= zero_out;
  end

  // The product is in [1,4): one bit of normalisation at most.
  wire        top     = p1_prod[47];
  wire [47:0] aligned = top ? p1_prod : {p1_prod[46:0], 1'b0};
  wire signed [10:0] exp_n = p1_exp + $signed({10'd0, top});

  wire [23:0] signif  = aligned[47:24];
  wire        rnd_bit = aligned[23];
  wire        sticky  = |aligned[22:0];
  wire        inc     = rnd_bit & (sticky | signif[0]);

  /* verilator lint_off UNUSEDSIGNAL */
  wire        round_cout;
  /* verilator lint_on UNUSEDSIGNAL */
  wire [24:0] signif_r;
  atlas_cpa #(.W(25)) u_round (
    .a({1'b0, signif}), .b(25'd0), .cin(inc), .sum(signif_r), .cout(round_cout)
  );
  wire        carry    = signif_r[24];
  /* verilator lint_off UNUSEDSIGNAL */
  wire [23:0] signif_f = carry ? signif_r[24:1] : signif_r[23:0];
  /* verilator lint_on UNUSEDSIGNAL */
  wire signed [10:0] exp_f = exp_n + $signed({10'd0, carry});

  logic [31:0] s1_result;
  always_comb begin
    if (p1_nan)                          s1_result = 32'h7FC00000;
    else if (p1_inf)                     s1_result = {p1_sign, 8'hFF, 23'd0};
    else if (p1_zero)                    s1_result = {p1_sign, 31'd0};
    else if (exp_f >= $signed(11'sd255)) s1_result = {p1_sign, 8'hFF, 23'd0};
    else if (exp_f <= 0)                 s1_result = {p1_sign, 31'd0};
    else                                 s1_result = {p1_sign, exp_f[7:0], signif_f[22:0]};
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= p1_valid;
  end

  always_ff @(posedge clk) begin
    result <= s1_result;
  end

endmodule
