//===========================================================================
// atlas_msb_idx -- index of the most significant set bit (tree structured)
//
// Normalising the accumulator needs the position of the leading one.  Written
// as a linear priority scan this becomes a W-deep chain of gates and dominates
// the normalise stage; built as a balanced tree it is log2(W) deep instead.
//
// Each node asks only "is there a one in my upper half?".  That bit is the
// next index bit, and it selects which half's index to carry up.
//
// W is padded internally to a power of two, so any width may be requested.
// Combinational.
//===========================================================================

module atlas_msb_idx #(
  parameter int unsigned W  = 34,
  parameter int unsigned IW = $clog2(W)     // width of the returned index
) (
  input  logic [W-1:0]    din,
  output logic [IW-1:0]   idx,
  output logic            nonzero
);

  localparam int unsigned WP = 1 << IW;     // padded width, a power of two

  logic [WP-1:0] padded;
  assign padded = {{(WP-W){1'b0}}, din};

  atlas_msb_idx_p2 #(.W(WP), .IW(IW)) u_tree (
    .din(padded), .idx(idx), .nonzero(nonzero)
  );

endmodule


//---------------------------------------------------------------------------
// Power-of-two recursion.  Kept separate so the padding above stays a single
// non-recursive wrapper.
//---------------------------------------------------------------------------
module atlas_msb_idx_p2 #(
  parameter int unsigned W  = 64,
  parameter int unsigned IW = 6
) (
  input  logic [W-1:0]  din,
  output logic [IW-1:0] idx,
  output logic          nonzero
);

  generate
    if (W == 2) begin : g_base
      assign nonzero = |din;
      assign idx     = din[1];               // upper bit wins
    end else begin : g_node
      logic [IW-2:0] idx_lo, idx_hi;
      logic          nz_lo,  nz_hi;

      atlas_msb_idx_p2 #(.W(W/2), .IW(IW-1)) u_lo (
        .din(din[W/2-1:0]),   .idx(idx_lo), .nonzero(nz_lo));
      atlas_msb_idx_p2 #(.W(W/2), .IW(IW-1)) u_hi (
        .din(din[W-1:W/2]),   .idx(idx_hi), .nonzero(nz_hi));

      assign nonzero = nz_hi | nz_lo;
      assign idx     = {nz_hi, nz_hi ? idx_hi : idx_lo};
    end
  endgenerate

endmodule
