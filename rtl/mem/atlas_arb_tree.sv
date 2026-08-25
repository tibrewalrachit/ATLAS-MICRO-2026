//===========================================================================
// atlas_arb_tree -- pick the valid entry with the smallest key
//
// A memory scheduler has to find the oldest ready request in its queue.
// Written as a sequential scan -- "if this one is valid and older than the
// best so far, take it" -- that becomes N chained compare-and-select stages,
// each a full key comparison.  With a 16-entry queue and 16-bit sequence
// numbers, that measured as a 5.7 ns path and was by far the worst in the
// design.
//
// A tournament tree does the same job in log2(N) levels.  Each node picks
// between its two children: the left one wins if it is valid and its key is no
// larger, otherwise the right one.  Ties go left, which keeps the result
// deterministic and matches the scan's behaviour.
//
// Keys are compared as unsigned.  A caller ordering by a wrapping sequence
// number should offset its keys so the comparison stays monotonic over the
// window it cares about.
//
// N must be a power of two.  Combinational.
//===========================================================================

module atlas_arb_tree #(
  parameter int unsigned N    = 16,
  parameter int unsigned KW   = 16,
  parameter int unsigned IW   = $clog2(N)
) (
  input  logic [N-1:0]     valid,
  input  logic [N*KW-1:0]  key,
  output logic             out_valid,
  output logic [IW-1:0]    out_idx,
  output logic [KW-1:0]    out_key
);

  generate
    // The recursion bottoms out at two entries, not one: a single-entry node
    // would need a zero-width index.
    if (N == 2) begin : g_leaf
      wire [KW-1:0] k0 = key[0      +: KW];
      wire [KW-1:0] k1 = key[KW     +: KW];
      wire take_lo = valid[0] & (~valid[1] | (k0 <= k1));
      assign out_valid = valid[0] | valid[1];
      assign out_idx   = ~take_lo;
      assign out_key   = take_lo ? k0 : k1;
    end else begin : g_node
      logic          lv, hv;
      logic [IW-2:0] li, hi;
      logic [KW-1:0] lk, hk;

      atlas_arb_tree #(.N(N/2), .KW(KW)) u_lo (
        .valid(valid[N/2-1:0]),
        .key  (key[0 +: (N/2)*KW]),
        .out_valid(lv), .out_idx(li), .out_key(lk)
      );
      atlas_arb_tree #(.N(N/2), .KW(KW)) u_hi (
        .valid(valid[N-1:N/2]),
        .key  (key[(N/2)*KW +: (N/2)*KW]),
        .out_valid(hv), .out_idx(hi), .out_key(hk)
      );

      wire take_lo = lv & (~hv | (lk <= hk));

      assign out_valid = lv | hv;
      assign out_idx   = take_lo ? {1'b0, li} : {1'b1, hi};
      assign out_key   = take_lo ? lk : hk;
    end
  endgenerate

endmodule
