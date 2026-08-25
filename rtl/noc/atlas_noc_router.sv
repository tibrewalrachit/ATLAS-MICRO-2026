//===========================================================================
// atlas_noc_router -- 5-port mesh router
//
// ATLAS's cloud chip puts its 16 cores on a 4x4 mesh with 128-byte flits
// (configs/architecture/chip/cloud/stratum/atlas.yaml, noc/4x4_mesh).  This is
// one node of that mesh: four neighbour ports plus the local core.
//
// Dimension-order (XY) routing: a packet travels in X until its column matches,
// then in Y.  XY is deadlock-free on a mesh without needing virtual channels
// to break cycles, which is why a mesh accelerator NoC almost always uses it --
// the turns it forbids (Y-then-X) are exactly the ones that would close a
// cycle in the channel dependency graph.
//
// Flow control is credit-based, per input buffer.  A flit is only sent when
// the downstream buffer is known to have room, so nothing is ever dropped and
// no retransmission path is needed.
//
// The traffic this carries is tensor parallel reductions and MoE expert
// dispatch, both of which move 128-byte lines; at that flit size a packet is
// usually one or two flits, so the router is single-flit-per-packet and needs
// no wormhole reassembly state.
//===========================================================================
`include "atlas_defs.svh"

module atlas_noc_router #(
  parameter int unsigned FLIT_W = 128,           // payload bits carried per port
  parameter int unsigned XW     = 2,             // mesh coordinate widths
  parameter int unsigned YW     = 2,
  parameter int unsigned DEPTH  = ATLAS_NOC_BUF_DEP,
  parameter int unsigned PORTS  = 5              // N, E, S, W, Local
) (
  input  logic                   clk,
  input  logic                   rst_n,

  input  logic [XW-1:0]          my_x,
  input  logic [YW-1:0]          my_y,

  // Per-port input channels
  input  logic [PORTS-1:0]              in_valid,
  input  logic [PORTS*FLIT_W-1:0]       in_flit,
  input  logic [PORTS*XW-1:0]           in_dx,     // destination coordinates
  input  logic [PORTS*YW-1:0]           in_dy,
  output logic [PORTS-1:0]              in_ready,

  // Per-port output channels
  output logic [PORTS-1:0]              out_valid,
  output logic [PORTS*FLIT_W-1:0]       out_flit,
  output logic [PORTS*XW-1:0]           out_dx,
  output logic [PORTS*YW-1:0]           out_dy,
  input  logic [PORTS-1:0]              out_ready
);

  localparam int unsigned P_N = 0, P_E = 1, P_S = 2, P_W = 3, P_L = 4;
  localparam int unsigned AW  = $clog2(DEPTH);
  localparam int unsigned ENT_W = FLIT_W + XW + YW;

  //=========================================================================
  // Input buffers, one FIFO per port
  //=========================================================================
  logic [ENT_W-1:0] fifo [0:PORTS-1][0:DEPTH-1];
  logic [AW:0]      wr_p [0:PORTS-1];
  logic [AW:0]      rd_p [0:PORTS-1];

  logic [PORTS-1:0] f_empty, f_full;
  logic [ENT_W-1:0] head [0:PORTS-1];

  integer fp;
  always_comb begin
    for (fp = 0; fp < PORTS; fp = fp + 1) begin
      f_empty[fp] = (wr_p[fp] == rd_p[fp]);
      f_full[fp]  = (wr_p[fp][AW-1:0] == rd_p[fp][AW-1:0]) &&
                    (wr_p[fp][AW] != rd_p[fp][AW]);
      head[fp]    = fifo[fp][rd_p[fp][AW-1:0]];
    end
  end

  assign in_ready = ~f_full;

  //=========================================================================
  // Route computation: XY, evaluated on each buffer's head flit
  //=========================================================================
  logic [PORTS-1:0][2:0] want;      // desired output port per input
  logic [PORTS-1:0]      req;

  integer rp;
  always_comb begin
    for (rp = 0; rp < PORTS; rp = rp + 1) begin
      automatic logic [XW-1:0] dx = head[rp][FLIT_W +: XW];
      automatic logic [YW-1:0] dy = head[rp][FLIT_W+XW +: YW];
      req[rp]  = ~f_empty[rp];
      if (dx != my_x)      want[rp] = (dx > my_x) ? 3'(P_E) : 3'(P_W);
      else if (dy != my_y) want[rp] = (dy > my_y) ? 3'(P_S) : 3'(P_N);
      else                 want[rp] = 3'(P_L);
    end
  end

  //=========================================================================
  // Output arbitration: round-robin per output port.
  //
  // A rotating priority pointer, rather than fixed priority, so a busy input
  // cannot starve its neighbours -- with fixed priority, north-bound traffic
  // through a corner router would lock out the local core indefinitely.
  //=========================================================================
  logic [2:0] rr_ptr [0:PORTS-1];

  logic [PORTS-1:0][2:0] grant_src;   // which input each output takes
  logic [PORTS-1:0]      grant_vld;
  logic [PORTS-1:0]      src_granted;

  integer o, s, si;
  always_comb begin
    src_granted = '0;
    for (o = 0; o < PORTS; o = o + 1) begin
      grant_vld[o] = 1'b0;
      grant_src[o] = 3'd0;
      for (s = 0; s < PORTS; s = s + 1) begin
        // Walk the inputs starting from this output's rotating pointer.
        si = (int'(rr_ptr[o]) + s) % PORTS;
        // An output can take a new flit only when its register is free:
        // either empty, or being consumed this cycle.  Granting on out_ready
        // alone would let a new flit overwrite one the neighbour has not yet
        // accepted, and the old flit would simply vanish.
        if (!grant_vld[o] && req[si] && (want[si] == 3'(o)) &&
            (!out_valid[o] || out_ready[o])) begin
          grant_vld[o] = 1'b1;
          grant_src[o] = 3'(si);
        end
      end
      if (grant_vld[o]) src_granted[grant_src[o]] = 1'b1;
    end
  end

  //=========================================================================
  // Output registers
  //=========================================================================
  integer q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (q = 0; q < PORTS; q = q + 1) begin
        wr_p[q]      <= '0;
        rd_p[q]      <= '0;
        rr_ptr[q]    <= 3'd0;
        out_valid[q] <= 1'b0;
      end
    end else begin
      // Enqueue
      for (q = 0; q < PORTS; q = q + 1) begin
        if (in_valid[q] && !f_full[q]) begin
          fifo[q][wr_p[q][AW-1:0]] <= {in_dy[q*YW +: YW], in_dx[q*XW +: XW],
                                       in_flit[q*FLIT_W +: FLIT_W]};
          wr_p[q] <= wr_p[q] + 1'b1;
        end
      end

      // Dequeue whichever inputs were granted
      for (q = 0; q < PORTS; q = q + 1)
        if (src_granted[q]) rd_p[q] <= rd_p[q] + 1'b1;

      // Drive outputs and advance the arbiter.  The register is only
      // disturbed when it is free, so a stalled flit is held, not dropped.
      for (q = 0; q < PORTS; q = q + 1) begin
        if (!out_valid[q] || out_ready[q]) out_valid[q] <= grant_vld[q];
        if (grant_vld[q]) begin
          out_flit[q*FLIT_W +: FLIT_W] <= head[grant_src[q]][FLIT_W-1:0];
          out_dx  [q*XW     +: XW]     <= head[grant_src[q]][FLIT_W    +: XW];
          out_dy  [q*YW     +: YW]     <= head[grant_src[q]][FLIT_W+XW +: YW];
          // Next time, start the search after the input just served.
          rr_ptr[q] <= (grant_src[q] == 3'(PORTS-1)) ? 3'd0 : (grant_src[q] + 3'd1);
        end
      end
    end
  end

endmodule
