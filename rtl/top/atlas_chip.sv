//===========================================================================
// atlas_chip -- the ATLAS accelerator die
//
// CORE_NUM cores on a MESH_X by MESH_Y mesh, each with its own HBDRAM channel
// bonded to the DRAM die above it.  The cloud configuration is 16 cores on a
// 4x4 mesh with 16 channels, which is the 1 TB/s design point of
// configs/architecture/dram/cloud/cloud_1TBps.yaml.
//
// One channel per core, rather than a shared memory system, is the defining
// choice of a hybrid-bonded design.  Bonding pitch gives enough connections to
// put a full 1024-bit channel under every core, so a core's weight stream never
// crosses the NoC -- the mesh carries only what is genuinely shared between
// cores (tensor-parallel reductions and MoE expert dispatch), and bandwidth to
// weights scales with core count instead of contending for a central port.
//
// The DRAM clock is half the core clock, generated here as an enable rather
// than a second clock domain: both come from the same PLL, so a divider and an
// enable are exact and cost no synchronisers.
//===========================================================================
`include "atlas_defs.svh"

module atlas_chip #(
  parameter int unsigned MESH_X    = ATLAS_NOC_X,
  parameter int unsigned MESH_Y    = ATLAS_NOC_Y,
  parameter int unsigned CORE_NUM  = MESH_X * MESH_Y,
  parameter int unsigned PE_ROWS   = ATLAS_PE_ROWS,
  parameter int unsigned PE_COLS   = ATLAS_PE_COLS,
  parameter int unsigned DOT_N     = ATLAS_DOT_N,
  parameter int unsigned ACC_N     = 4,
  parameter int unsigned VEC_N     = 32,
  parameter int unsigned SFU_RATIO = 8,
  parameter int unsigned DQ_W      = ATLAS_HBD_DQ_W,
  parameter int unsigned BUF_AW    = 20,
  parameter int unsigned ACC_W     = $clog2(ACC_N),
  parameter int unsigned XW        = $clog2(MESH_X),
  parameter int unsigned YW        = $clog2(MESH_Y),
  parameter int unsigned NSFU      = (VEC_N + SFU_RATIO - 1) / SFU_RATIO
) (
  input  logic clk,
  input  logic rst_n,

  // ---- per-core matrix descriptors ----
  input  logic [CORE_NUM-1:0]                    mm_valid,
  input  logic [CORE_NUM*2-1:0]                  mm_fmt_a,
  input  logic [CORE_NUM*2-1:0]                  mm_fmt_b,
  input  logic [CORE_NUM*ACC_W-1:0]              mm_acc_id,
  input  logic [CORE_NUM-1:0]                    mm_first,
  input  logic [CORE_NUM-1:0]                    mm_last,
  input  logic [CORE_NUM*PE_ROWS*DOT_N*8-1:0]    mm_a,
  input  logic [CORE_NUM*PE_ROWS*8-1:0]          mm_a_scale,
  input  logic [CORE_NUM*PE_COLS*DOT_N*8-1:0]    mm_b,
  input  logic [CORE_NUM*PE_COLS*8-1:0]          mm_b_scale,
  output logic [CORE_NUM*PE_ROWS*PE_COLS-1:0]    mm_out_valid,
  output logic [CORE_NUM*PE_ROWS*PE_COLS*32-1:0] mm_out,

  // ---- per-core vector port ----
  input  logic [CORE_NUM-1:0]           vec_valid,
  input  logic [CORE_NUM*3-1:0]         vec_op,
  input  logic [CORE_NUM*VEC_N*32-1:0]  vec_a,
  input  logic [CORE_NUM*VEC_N*32-1:0]  vec_b,
  output logic [CORE_NUM-1:0]           vec_out_valid,
  output logic [CORE_NUM*VEC_N*32-1:0]  vec_y,
  input  logic [CORE_NUM-1:0]           sfu_valid,
  input  logic [CORE_NUM*2-1:0]         sfu_op,
  input  logic [CORE_NUM*NSFU*32-1:0]   sfu_x,
  output logic [CORE_NUM-1:0]           sfu_out_valid,
  output logic [CORE_NUM*NSFU*32-1:0]   sfu_y,

  // ---- per-core DMA descriptors ----
  input  logic [CORE_NUM-1:0]                    dma_valid,
  output logic [CORE_NUM-1:0]                    dma_ready,
  input  logic [CORE_NUM-1:0]                    dma_store,
  input  logic [CORE_NUM*ATLAS_HBD_ROW_W-1:0]    dma_row,
  input  logic [CORE_NUM*ATLAS_HBD_COL_W-1:0]    dma_col,
  input  logic [CORE_NUM*BUF_AW-1:0]             dma_buf_addr,
  input  logic [CORE_NUM*16-1:0]                 dma_lines,
  output logic [CORE_NUM-1:0]                    dma_done,

  // ---- HBDRAM: one channel per core, to the bonded die ----
  output logic [CORE_NUM*3-1:0]                  dram_cmd,
  output logic [CORE_NUM*ATLAS_HBD_ROW_W-1:0]    dram_row,
  output logic [CORE_NUM*ATLAS_HBD_COL_W-1:0]    dram_col,
  output logic [CORE_NUM*DQ_W-1:0]               dram_wdata,
  input  logic [CORE_NUM*DQ_W-1:0]               dram_rdata,
  input  logic [CORE_NUM-1:0]                    dram_rvalid,

  // ---- statistics ----
  output logic [CORE_NUM*32-1:0]                 stat_row_hit,
  output logic [CORE_NUM*32-1:0]                 stat_row_miss,
  output logic [CORE_NUM*32-1:0]                 stat_refresh
);

  //-------------------------------------------------------------------------
  // HBDRAM runs at 500 MT/s against a 1 GHz core clock: one tick every two
  // cycles, from the same PLL, so this is an enable and not a clock domain.
  //-------------------------------------------------------------------------
  logic dram_tick;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) dram_tick <= 1'b0;
    else        dram_tick <= ~dram_tick;
  end

  //-------------------------------------------------------------------------
  // Mesh
  //-------------------------------------------------------------------------
  logic [CORE_NUM-1:0]                inj_valid, inj_ready, ej_valid, ej_ready;
  logic [CORE_NUM*ATLAS_FLIT_W-1:0]   inj_flit, ej_flit;
  logic [CORE_NUM*XW-1:0]             inj_dx;
  logic [CORE_NUM*YW-1:0]             inj_dy;

  assign inj_dx = '0;
  assign inj_dy = '0;

  atlas_noc_mesh #(
    .MESH_X(MESH_X), .MESH_Y(MESH_Y), .FLIT_W(ATLAS_FLIT_W)
  ) u_mesh (
    .clk(clk), .rst_n(rst_n),
    .inj_valid(inj_valid), .inj_flit(inj_flit),
    .inj_dx(inj_dx), .inj_dy(inj_dy), .inj_ready(inj_ready),
    .ej_valid(ej_valid), .ej_flit(ej_flit), .ej_ready(ej_ready)
  );

  //-------------------------------------------------------------------------
  // Cores
  //-------------------------------------------------------------------------
  genvar c;
  generate
    for (c = 0; c < CORE_NUM; c++) begin : g_core
      atlas_core #(
        .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS), .DOT_N(DOT_N), .ACC_N(ACC_N),
        .VEC_N(VEC_N), .SFU_RATIO(SFU_RATIO), .DQ_W(DQ_W), .BUF_AW(BUF_AW)
      ) u_core (
        .clk(clk), .rst_n(rst_n), .dram_tick(dram_tick),

        .mm_valid  (mm_valid[c]),
        .mm_fmt_a  (mm_fmt_a[c*2 +: 2]),
        .mm_fmt_b  (mm_fmt_b[c*2 +: 2]),
        .mm_acc_id (mm_acc_id[c*ACC_W +: ACC_W]),
        .mm_first  (mm_first[c]),
        .mm_last   (mm_last[c]),
        .mm_a      (mm_a[c*PE_ROWS*DOT_N*8 +: PE_ROWS*DOT_N*8]),
        .mm_a_scale(mm_a_scale[c*PE_ROWS*8 +: PE_ROWS*8]),
        .mm_b      (mm_b[c*PE_COLS*DOT_N*8 +: PE_COLS*DOT_N*8]),
        .mm_b_scale(mm_b_scale[c*PE_COLS*8 +: PE_COLS*8]),
        .mm_out_valid(mm_out_valid[c*PE_ROWS*PE_COLS +: PE_ROWS*PE_COLS]),
        .mm_out      (mm_out[c*PE_ROWS*PE_COLS*32 +: PE_ROWS*PE_COLS*32]),

        .vec_valid    (vec_valid[c]),
        .vec_op       (vec_op[c*3 +: 3]),
        .vec_a        (vec_a[c*VEC_N*32 +: VEC_N*32]),
        .vec_b        (vec_b[c*VEC_N*32 +: VEC_N*32]),
        .vec_out_valid(vec_out_valid[c]),
        .vec_y        (vec_y[c*VEC_N*32 +: VEC_N*32]),
        .sfu_valid    (sfu_valid[c]),
        .sfu_op       (sfu_op[c*2 +: 2]),
        .sfu_x        (sfu_x[c*NSFU*32 +: NSFU*32]),
        .sfu_out_valid(sfu_out_valid[c]),
        .sfu_y        (sfu_y[c*NSFU*32 +: NSFU*32]),

        .dma_valid   (dma_valid[c]),
        .dma_ready   (dma_ready[c]),
        .dma_store   (dma_store[c]),
        .dma_row     (dma_row[c*ATLAS_HBD_ROW_W +: ATLAS_HBD_ROW_W]),
        .dma_col     (dma_col[c*ATLAS_HBD_COL_W +: ATLAS_HBD_COL_W]),
        .dma_buf_addr(dma_buf_addr[c*BUF_AW +: BUF_AW]),
        .dma_lines   (dma_lines[c*16 +: 16]),
        .dma_done    (dma_done[c]),

        .dram_cmd   (dram_cmd[c*3 +: 3]),
        .dram_row   (dram_row[c*ATLAS_HBD_ROW_W +: ATLAS_HBD_ROW_W]),
        .dram_col   (dram_col[c*ATLAS_HBD_COL_W +: ATLAS_HBD_COL_W]),
        .dram_wdata (dram_wdata[c*DQ_W +: DQ_W]),
        .dram_rdata (dram_rdata[c*DQ_W +: DQ_W]),
        .dram_rvalid(dram_rvalid[c]),

        .noc_inj_valid(inj_valid[c]),
        .noc_inj_flit (inj_flit[c*ATLAS_FLIT_W +: ATLAS_FLIT_W]),
        .noc_inj_ready(inj_ready[c]),
        .noc_ej_valid (ej_valid[c]),
        .noc_ej_flit  (ej_flit[c*ATLAS_FLIT_W +: ATLAS_FLIT_W]),
        .noc_ej_ready (ej_ready[c]),

        .stat_row_hit (stat_row_hit[c*32 +: 32]),
        .stat_row_miss(stat_row_miss[c*32 +: 32]),
        .stat_refresh (stat_refresh[c*32 +: 32])
      );
    end
  endgenerate

endmodule
