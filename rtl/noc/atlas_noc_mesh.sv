//===========================================================================
// atlas_noc_mesh -- MESH_X by MESH_Y mesh of atlas_noc_router
//
// The cloud ATLAS chip is a 4x4 mesh of 16 cores.  This wires the routers into
// a grid and exposes each node's local port.
//
// Edge ports are tied off rather than wrapped into a torus: XY routing on a
// mesh never sends a packet off the edge, because a packet only moves in a
// direction that reduces the distance to its destination, and no destination
// lies outside the grid.  Wrapping would also break XY's deadlock freedom.
//===========================================================================
`include "atlas_defs.svh"

module atlas_noc_mesh #(
  parameter int unsigned MESH_X = ATLAS_NOC_X,
  parameter int unsigned MESH_Y = ATLAS_NOC_Y,
  parameter int unsigned FLIT_W = 128,
  parameter int unsigned XW     = $clog2(MESH_X),
  parameter int unsigned YW     = $clog2(MESH_Y),
  parameter int unsigned NODES  = MESH_X * MESH_Y
) (
  input  logic                      clk,
  input  logic                      rst_n,

  // Local injection, one per node (node index = y*MESH_X + x)
  input  logic [NODES-1:0]          inj_valid,
  input  logic [NODES*FLIT_W-1:0]   inj_flit,
  input  logic [NODES*XW-1:0]       inj_dx,
  input  logic [NODES*YW-1:0]       inj_dy,
  output logic [NODES-1:0]          inj_ready,

  // Local ejection
  output logic [NODES-1:0]          ej_valid,
  output logic [NODES*FLIT_W-1:0]   ej_flit,
  input  logic [NODES-1:0]          ej_ready
);

  localparam int unsigned P_N = 0, P_E = 1, P_S = 2, P_W = 3, P_L = 4;
  localparam int unsigned PORTS = 5;

  // Per-node port bundles, held as flat packed vectors: yosys' Verilog front
  // end does not accept multi-dimensional packed declarations, and these
  // sources are read by Verilator, Icarus and yosys unmodified.
  //
  // Edge routers keep the full five-port bundle but their outward-facing ports
  // connect to nothing, so those bits are legitimately unread.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [NODES*PORTS-1:0]        iv, ir, ov, orr;
  logic [NODES*PORTS*FLIT_W-1:0] ifl, ofl;
  logic [NODES*PORTS*XW-1:0]     idx, odx;
  logic [NODES*PORTS*YW-1:0]     idy, ody;
  /* verilator lint_on UNUSEDSIGNAL */

  genvar gx, gy;
  generate
    for (gy = 0; gy < MESH_Y; gy++) begin : g_row
      for (gx = 0; gx < MESH_X; gx++) begin : g_col
        localparam int unsigned NID = gy*MESH_X + gx;

        atlas_noc_router #(.FLIT_W(FLIT_W), .XW(XW), .YW(YW)) u_rt (
          .clk(clk), .rst_n(rst_n),
          .my_x(XW'(gx)), .my_y(YW'(gy)),
          .in_valid(iv[NID*PORTS +: PORTS]),
          .in_flit(ifl[NID*PORTS*FLIT_W +: PORTS*FLIT_W]),
          .in_dx(idx[NID*PORTS*XW +: PORTS*XW]),
          .in_dy(idy[NID*PORTS*YW +: PORTS*YW]),
          .in_ready(ir[NID*PORTS +: PORTS]),
          .out_valid(ov[NID*PORTS +: PORTS]),
          .out_flit(ofl[NID*PORTS*FLIT_W +: PORTS*FLIT_W]),
          .out_dx(odx[NID*PORTS*XW +: PORTS*XW]),
          .out_dy(ody[NID*PORTS*YW +: PORTS*YW]),
          .out_ready(orr[NID*PORTS +: PORTS])
        );

        // ---- local port ----
        assign iv[NID*PORTS + P_L]                 = inj_valid[NID];
        assign ifl[(NID*PORTS + P_L)*FLIT_W +: FLIT_W] = inj_flit[NID*FLIT_W +: FLIT_W];
        assign idx[(NID*PORTS + P_L)*XW +: XW]        = inj_dx[NID*XW +: XW];
        assign idy[(NID*PORTS + P_L)*YW +: YW]        = inj_dy[NID*YW +: YW];
        assign inj_ready[NID]                = ir[NID*PORTS + P_L];

        assign ej_valid[NID]                 = ov[NID*PORTS + P_L];
        assign ej_flit[NID*FLIT_W +: FLIT_W] = ofl[(NID*PORTS + P_L)*FLIT_W +: FLIT_W];
        assign orr[NID*PORTS + P_L]                 = ej_ready[NID];

        // ---- north neighbour ----
        if (gy > 0) begin : g_north
          localparam int unsigned UP = NID - MESH_X;
          assign iv[NID*PORTS + P_N]                  = ov[UP*PORTS + P_S];
          assign ifl[(NID*PORTS + P_N)*FLIT_W +: FLIT_W] = ofl[(UP*PORTS + P_S)*FLIT_W +: FLIT_W];
          assign idx[(NID*PORTS + P_N)*XW +: XW]         = odx[(UP*PORTS + P_S)*XW +: XW];
          assign idy[(NID*PORTS + P_N)*YW +: YW]         = ody[(UP*PORTS + P_S)*YW +: YW];
          assign orr[NID*PORTS + P_N]                  = ir[UP*PORTS + P_S];
        end else begin : g_north_edge
          assign iv[NID*PORTS + P_N]                  = 1'b0;
          assign ifl[(NID*PORTS + P_N)*FLIT_W +: FLIT_W] = '0;
          assign idx[(NID*PORTS + P_N)*XW +: XW]         = '0;
          assign idy[(NID*PORTS + P_N)*YW +: YW]         = '0;
          assign orr[NID*PORTS + P_N]                  = 1'b1;
        end

        // ---- south neighbour ----
        if (gy < MESH_Y-1) begin : g_south
          localparam int unsigned DN = NID + MESH_X;
          assign iv[NID*PORTS + P_S]                  = ov[DN*PORTS + P_N];
          assign ifl[(NID*PORTS + P_S)*FLIT_W +: FLIT_W] = ofl[(DN*PORTS + P_N)*FLIT_W +: FLIT_W];
          assign idx[(NID*PORTS + P_S)*XW +: XW]         = odx[(DN*PORTS + P_N)*XW +: XW];
          assign idy[(NID*PORTS + P_S)*YW +: YW]         = ody[(DN*PORTS + P_N)*YW +: YW];
          assign orr[NID*PORTS + P_S]                  = ir[DN*PORTS + P_N];
        end else begin : g_south_edge
          assign iv[NID*PORTS + P_S]                  = 1'b0;
          assign ifl[(NID*PORTS + P_S)*FLIT_W +: FLIT_W] = '0;
          assign idx[(NID*PORTS + P_S)*XW +: XW]         = '0;
          assign idy[(NID*PORTS + P_S)*YW +: YW]         = '0;
          assign orr[NID*PORTS + P_S]                  = 1'b1;
        end

        // ---- west neighbour ----
        if (gx > 0) begin : g_west
          localparam int unsigned LF = NID - 1;
          assign iv[NID*PORTS + P_W]                  = ov[LF*PORTS + P_E];
          assign ifl[(NID*PORTS + P_W)*FLIT_W +: FLIT_W] = ofl[(LF*PORTS + P_E)*FLIT_W +: FLIT_W];
          assign idx[(NID*PORTS + P_W)*XW +: XW]         = odx[(LF*PORTS + P_E)*XW +: XW];
          assign idy[(NID*PORTS + P_W)*YW +: YW]         = ody[(LF*PORTS + P_E)*YW +: YW];
          assign orr[NID*PORTS + P_W]                  = ir[LF*PORTS + P_E];
        end else begin : g_west_edge
          assign iv[NID*PORTS + P_W]                  = 1'b0;
          assign ifl[(NID*PORTS + P_W)*FLIT_W +: FLIT_W] = '0;
          assign idx[(NID*PORTS + P_W)*XW +: XW]         = '0;
          assign idy[(NID*PORTS + P_W)*YW +: YW]         = '0;
          assign orr[NID*PORTS + P_W]                  = 1'b1;
        end

        // ---- east neighbour ----
        if (gx < MESH_X-1) begin : g_east
          localparam int unsigned RT = NID + 1;
          assign iv[NID*PORTS + P_E]                  = ov[RT*PORTS + P_W];
          assign ifl[(NID*PORTS + P_E)*FLIT_W +: FLIT_W] = ofl[(RT*PORTS + P_W)*FLIT_W +: FLIT_W];
          assign idx[(NID*PORTS + P_E)*XW +: XW]         = odx[(RT*PORTS + P_W)*XW +: XW];
          assign idy[(NID*PORTS + P_E)*YW +: YW]         = ody[(RT*PORTS + P_W)*YW +: YW];
          assign orr[NID*PORTS + P_E]                  = ir[RT*PORTS + P_W];
        end else begin : g_east_edge
          assign iv[NID*PORTS + P_E]                  = 1'b0;
          assign ifl[(NID*PORTS + P_E)*FLIT_W +: FLIT_W] = '0;
          assign idx[(NID*PORTS + P_E)*XW +: XW]         = '0;
          assign idy[(NID*PORTS + P_E)*YW +: YW]         = '0;
          assign orr[NID*PORTS + P_E]                  = 1'b1;
        end
      end
    end
  endgenerate

endmodule
