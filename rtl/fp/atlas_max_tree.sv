//===========================================================================
// atlas_max_tree -- balanced signed maximum reduction over N packed elements
//
// Used to find the largest product exponent in an MX block, which sets the
// alignment reference for the accumulator.  Written as a recursive module so
// the tree is structurally balanced (log2(N) comparator delays) and so no
// shared storage vector appears that a tool could mistake for a combinational
// loop.
//
// N must be a power of two.  Combinational.
//===========================================================================

module atlas_max_tree #(
  parameter int unsigned N = 32,
  parameter int unsigned W = 7
) (
  input  logic [N*W-1:0]      din,   // N signed elements, element i at [i*W +: W]
  output logic signed [W-1:0] dmax
);

  generate
    if (N == 1) begin : g_leaf
      assign dmax = $signed(din);
    end else begin : g_node
      logic signed [W-1:0] lo, hi;
      atlas_max_tree #(.N(N/2), .W(W)) u_lo (
        .din (din[0            +: (N/2)*W]), .dmax(lo));
      atlas_max_tree #(.N(N/2), .W(W)) u_hi (
        .din (din[(N/2)*W      +: (N/2)*W]), .dmax(hi));
      assign dmax = (lo > hi) ? lo : hi;
    end
  endgenerate

endmodule
