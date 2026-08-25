//===========================================================================
// atlas_mul_csa -- unsigned W x W multiplier, carry-save reduction
//
// Written out structurally rather than as `a * b` for the same reason the
// dot-product engine's reduction is: yosys expands a multiply into a
// ripple-carry array before ABC sees it, and ABC's rewriting works on small
// windows, so it cannot recover a fast structure afterwards.  A 24x24 array
// built that way is the critical path of every block that scales a vector.
//
// Partial products are reduced to a (sum, carry) pair by the same carry-save
// tree the MAC array uses, leaving exactly one carry propagation at the end.
// Depth is 2*log2(N/2) full-adder delays plus one adder, against roughly 2W
// for the array form.
//
// The partial-product count is rounded up to a power of two with zero rows,
// which the tree optimises away.
//===========================================================================

module atlas_mul_csa #(
  parameter int unsigned W = 24
) (
  input  logic [W-1:0]     a,
  input  logic [W-1:0]     b,
  output logic [2*W-1:0]   p
);

  localparam int unsigned PW = 2*W;
  localparam int unsigned NP = 1 << $clog2(W);   // padded partial-product count

  logic [NP*PW-1:0] pp;

  genvar i;
  generate
    for (i = 0; i < NP; i++) begin : g_pp
      if (i < W) begin : g_real
        assign pp[i*PW +: PW] = b[i] ? (PW'(a) << i) : {PW{1'b0}};
      end else begin : g_pad
        assign pp[i*PW +: PW] = {PW{1'b0}};
      end
    end
  endgenerate

  logic [PW-1:0] s_v, c_v;
  atlas_csa_tree #(.N(NP), .W(PW)) u_csa (
    .din(pp), .s_out(s_v), .c_out(c_v)
  );

  // The product fits in PW bits, so the tree's modulo-2^PW result is exact and
  // this carry-out is never meaningful.
  /* verilator lint_off UNUSEDSIGNAL */
  logic cout;
  /* verilator lint_on UNUSEDSIGNAL */

  atlas_cpa #(.W(PW)) u_cpa (
    .a(s_v), .b(c_v), .cin(1'b0), .sum(p), .cout(cout)
  );

endmodule
