//===========================================================================
// atlas_csa_tree -- carry-save reduction of N terms to a (sum, carry) pair
//
// A plain adder tree over 32 terms puts five carry propagations in series,
// which is what dominates the delay of a wide MAC reduction.  This reduces
// the same 32 terms to two vectors using only 3:2 compressors, so no carry
// travels further than one bit until a single final adder resolves the pair:
//
//     sum(din[0..N-1])  ==  s_out + c_out      (mod 2^W)
//
// Every input must already be sign-extended to W bits.  All arithmetic is
// two's complement modulo 2^W, so the carry vectors are free to overflow --
// the result is still exact provided the true sum is representable in W bits,
// which the caller guarantees by sizing W as ALIGN_W + log2(N) + 1.
//
// Built recursively: each node reduces its two children's four vectors back
// to two with a 4:2 compressor (two 3:2 levels), giving a depth of
// 2*log2(N/2) full-adder delays and no ripple at all.
//
// N must be a power of two.  Combinational.
//===========================================================================

module atlas_csa_tree #(
  parameter int unsigned N = 32,
  parameter int unsigned W = 34
) (
  input  logic [N*W-1:0] din,
  output logic [W-1:0]   s_out,
  output logic [W-1:0]   c_out
);

  generate
    if (N == 1) begin : g_one
      assign s_out = din[W-1:0];
      assign c_out = {W{1'b0}};
    end else if (N == 2) begin : g_two
      // Two vectors already are a carry-save pair; nothing to compress.
      assign s_out = din[0   +: W];
      assign c_out = din[W   +: W];
    end else begin : g_node
      logic [W-1:0] s1, c1, s2, c2;

      atlas_csa_tree #(.N(N/2), .W(W)) u_lo (
        .din(din[0       +: (N/2)*W]), .s_out(s1), .c_out(c1));
      atlas_csa_tree #(.N(N/2), .W(W)) u_hi (
        .din(din[(N/2)*W +: (N/2)*W]), .s_out(s2), .c_out(c2));

      // 4:2 compression of {s1, c1, s2, c2} into {s_out, c_out}.
      // Stage A: s1 + c1 + s2  ==  sa + (ca << 1)
      // The top bit of each carry vector is shifted out below.  That is the
      // modulo-2^W behaviour described in the header, not a lost term.
      /* verilator lint_off UNUSEDSIGNAL */
      logic [W-1:0] sa, ca;
      assign sa = s1 ^ c1 ^ s2;
      assign ca = (s1 & c1) | (c1 & s2) | (s1 & s2);

      // Stage B: sa + (ca << 1) + c2  ==  sb + (cb << 1)
      logic [W-1:0] ca_s, sb, cb;
      /* verilator lint_on UNUSEDSIGNAL */
      assign ca_s = {ca[W-2:0], 1'b0};
      assign sb   = sa ^ ca_s ^ c2;
      assign cb   = (sa & ca_s) | (ca_s & c2) | (sa & c2);

      assign s_out = sb;
      assign c_out = {cb[W-2:0], 1'b0};
    end
  endgenerate

endmodule
