//===========================================================================
// atlas_cpa -- carry-select carry-propagate adder
//
// The carry-save tree leaves exactly one true addition in the reduction, and
// with a ripple adder that single add became the block's critical path: the
// 34-bit carry chain synthesised to roughly ninety gates in series.
//
// This is a recursive carry-select adder.  The upper half is evaluated for
// both possible incoming carries while the lower half is still resolving, so
// the carry crosses each split through one multiplexer instead of rippling.
// Depth is a BASE-wide ripple plus log2(W/BASE) mux levels.
//
// The cost is area: each split triples rather than doubles, giving
// O(W^1.58) instead of O(W).  That is affordable because there is only one of
// these per dot-product engine, against a 32-lane carry-save array.
//
// Combinational.  Any width; BASE sets the ripple leaf size.
//===========================================================================

module atlas_cpa #(
  parameter int unsigned W    = 34,
  parameter int unsigned BASE = 4
) (
  input  logic [W-1:0] a,
  input  logic [W-1:0] b,
  input  logic         cin,
  output logic [W-1:0] sum,
  output logic         cout
);

  generate
    if (W <= BASE) begin : g_leaf
      logic [W:0] r;
      assign r    = {1'b0, a} + {1'b0, b} + {{W{1'b0}}, cin};
      assign sum  = r[W-1:0];
      assign cout = r[W];
    end else begin : g_split
      localparam int unsigned LO = W/2;
      localparam int unsigned HI = W - LO;

      logic [LO-1:0] s_lo;
      logic          c_lo;
      atlas_cpa #(.W(LO), .BASE(BASE)) u_lo (
        .a(a[LO-1:0]), .b(b[LO-1:0]), .cin(cin), .sum(s_lo), .cout(c_lo));

      // Both halves of the speculation run concurrently with u_lo.
      logic [HI-1:0] s_hi0, s_hi1;
      logic          c_hi0, c_hi1;
      atlas_cpa #(.W(HI), .BASE(BASE)) u_hi0 (
        .a(a[W-1:LO]), .b(b[W-1:LO]), .cin(1'b0), .sum(s_hi0), .cout(c_hi0));
      atlas_cpa #(.W(HI), .BASE(BASE)) u_hi1 (
        .a(a[W-1:LO]), .b(b[W-1:LO]), .cin(1'b1), .sum(s_hi1), .cout(c_hi1));

      assign sum  = {(c_lo ? s_hi1 : s_hi0), s_lo};
      assign cout = c_lo ? c_hi1 : c_hi0;
    end
  endgenerate

endmodule
