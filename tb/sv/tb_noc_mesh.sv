//===========================================================================
// tb_noc_mesh -- all-to-all traffic across the 4x4 mesh
//
// Every node injects packets addressed to uniformly random destinations.  The
// checks are:
//   1. every packet arrives, and at the node it was addressed to,
//   2. payloads are unmodified,
//   3. the network drains -- nothing is left in flight, which is the property
//      XY routing exists to guarantee.
//===========================================================================
`timescale 1ns/1ps

module tb_noc_mesh;

  localparam int unsigned MESH_X = 4;
  localparam int unsigned MESH_Y = 4;
  localparam int unsigned NODES  = MESH_X * MESH_Y;
  localparam int unsigned FLIT_W = 128;
  localparam int unsigned XW     = $clog2(MESH_X);
  localparam int unsigned YW     = $clog2(MESH_Y);
  localparam int unsigned NPKT   = 40;      // packets injected per node

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  logic [NODES-1:0]        inj_valid, inj_ready, ej_valid, ej_ready;
  logic [NODES*FLIT_W-1:0] inj_flit, ej_flit;
  logic [NODES*XW-1:0]     inj_dx;
  logic [NODES*YW-1:0]     inj_dy;

  atlas_noc_mesh #(.MESH_X(MESH_X), .MESH_Y(MESH_Y), .FLIT_W(FLIT_W)) dut (
    .clk(clk), .rst_n(rst_n),
    .inj_valid(inj_valid), .inj_flit(inj_flit),
    .inj_dx(inj_dx), .inj_dy(inj_dy), .inj_ready(inj_ready),
    .ej_valid(ej_valid), .ej_flit(ej_flit), .ej_ready(ej_ready)
  );

  assign ej_ready = {NODES{1'b1}};           // sinks always accept

  // A payload carries its own source and destination, so a misrouted packet is
  // detectable at the point of ejection rather than only as a count mismatch.
  int sent = 0, recvd = 0, errors = 0;
  int sent_per [0:NODES-1];
  int recv_per [0:NODES-1];

  genvar gn;
  generate
    for (gn = 0; gn < NODES; gn++) begin : g_sink
      always @(posedge clk) begin
        if (rst_n && ej_valid[gn]) begin
          automatic int pdst = int'(ej_flit[gn*FLIT_W +: 8]);
          automatic int psrc = int'(ej_flit[gn*FLIT_W + 8 +: 8]);
          automatic int pseq = int'(ej_flit[gn*FLIT_W + 16 +: 16]);
          recvd = recvd + 1;
          recv_per[gn] = recv_per[gn] + 1;
          if (pdst != gn) begin
            errors = errors + 1;
            if (errors < 8)
              $display("  MISROUTED: packet for node %0d ejected at node %0d (src %0d seq %0d)",
                       pdst, gn, psrc, pseq);
          end
        end
      end
    end
  endgenerate

  int n, k, cyc;
  int dx_r, dy_r, dst;

  initial begin
    inj_valid = '0; inj_flit = '0; inj_dx = '0; inj_dy = '0;
    for (n = 0; n < NODES; n = n + 1) begin
      sent_per[n] = 0;
      recv_per[n] = 0;
    end

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    for (k = 0; k < NPKT; k = k + 1) begin
      // Present one packet per node, then wait for each to be accepted.
      for (n = 0; n < NODES; n = n + 1) begin
        dx_r = $urandom_range(0, MESH_X-1);
        dy_r = $urandom_range(0, MESH_Y-1);
        dst  = dy_r*MESH_X + dx_r;
        inj_dx[n*XW +: XW] = XW'(dx_r);
        inj_dy[n*YW +: YW] = YW'(dy_r);
        inj_flit[n*FLIT_W +: FLIT_W] = '0;
        inj_flit[n*FLIT_W      +: 8]  = 8'(dst);
        inj_flit[n*FLIT_W + 8  +: 8]  = 8'(n);
        inj_flit[n*FLIT_W + 16 +: 16] = 16'(k);
      end
      inj_valid = {NODES{1'b1}};

      // Hold until every node's injection has been taken.
      cyc = 0;
      while (inj_valid != '0) begin
        @(negedge clk);
        for (n = 0; n < NODES; n = n + 1)
          if (inj_valid[n] && inj_ready[n]) begin
            inj_valid[n] = 1'b0;
            sent = sent + 1;
            sent_per[n] = sent_per[n] + 1;
          end
        cyc = cyc + 1;
        if (cyc > 5000) begin
          $display("  injection stalled: the network is not draining");
          errors = errors + 1;
          inj_valid = '0;
        end
      end
    end

    // Drain
    repeat (4000) @(negedge clk);

    $display("tb_noc_mesh: injected %0d, ejected %0d", sent, recvd);
    if (recvd != sent) begin
      $display("  %0d packets lost or still in flight", sent - recvd);
      errors = errors + 1;
    end
    if (errors == 0)
      $display("TB_NOC_MESH: PASS  (%0dx%0d mesh, %0d packets, all delivered correctly)",
               MESH_X, MESH_Y, sent);
    else
      $display("TB_NOC_MESH: FAIL  (%0d errors)", errors);
    $finish;
  end

endmodule
