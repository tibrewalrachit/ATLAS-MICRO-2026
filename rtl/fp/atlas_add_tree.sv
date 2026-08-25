//===========================================================================
// atlas_add_tree -- balanced two's-complement adder tree over N elements
//
// The result width grows by exactly one bit per level (W + log2(N)), so no
// term can overflow and each level's adder is only as wide as it needs to be.
// Recursive construction keeps the tree balanced and avoids the shared-vector
// pattern that makes lint tools report a false combinational loop.
//
// N must be a power of two.  Combinational.
//===========================================================================

module atlas_add_tree #(
  parameter int unsigned N = 32,
  parameter int unsigned W = 29
) (
  input  logic [N*W-1:0]                     din,   // N signed elements
  output logic signed [W+$clog2(N)-1:0]      dsum
);

  localparam int unsigned LOG2N = $clog2(N);

  generate
    if (N == 1) begin : g_leaf
      assign dsum = $signed(din);
    end else begin : g_node
      // Each half produces W + log2(N/2) = W + LOG2N - 1 bits.
      localparam int unsigned HW = W + LOG2N - 1;
      logic signed [HW-1:0] lo, hi;
      atlas_add_tree #(.N(N/2), .W(W)) u_lo (
        .din (din[0       +: (N/2)*W]), .dsum(lo));
      atlas_add_tree #(.N(N/2), .W(W)) u_hi (
        .din (din[(N/2)*W +: (N/2)*W]), .dsum(hi));
      // Context width is W+LOG2N, so both operands sign-extend by one bit.
      assign dsum = lo + hi;
    end
  endgenerate

endmodule
