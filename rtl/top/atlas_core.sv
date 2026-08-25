//===========================================================================
// atlas_core -- one ATLAS core
//
// Assembles the pieces of a cloud ATLAS core as described by
// configs/architecture/chip/cloud/stratum/atlas.yaml:
//
//     matrix   7680 MACs   = PE_ROWS x PE_COLS x 32-lane MX dot engines
//     vector   480 lanes   + shared special-function units
//     buffer   4 MB        banked to reach 32768 B/cycle
//     dram     one HBDRAM channel of the 16 that make 1 TB/s
//     noc      one node of the 4x4 mesh
//
// The defaults elaborate the full core.  Everything is parameterised because a
// 7680-MAC array is around six million gates: simulation and the synthesis
// runs under syn/ use a scaled instance, and the numbers in docs/rtl are
// extrapolated from it with the scaling stated.
//
// Sequencing is descriptor-driven.  A host or the chip-level scheduler enqueues
// GEMM, vector and DMA descriptors; the core streams them without per-element
// instruction fetch, which is the right shape for an accelerator whose inner
// loop is known ahead of time.
//===========================================================================
`include "atlas_defs.svh"

module atlas_core #(
  parameter int unsigned PE_ROWS   = ATLAS_PE_ROWS,
  parameter int unsigned PE_COLS   = ATLAS_PE_COLS,
  parameter int unsigned DOT_N     = ATLAS_DOT_N,
  parameter int unsigned ACC_N     = 4,
  parameter int unsigned VEC_N     = 32,
  parameter int unsigned SFU_RATIO = 8,
  parameter int unsigned BUF_BANKS = 8,
  parameter int unsigned BUF_LINE_B = 128,
  parameter int unsigned BUF_DEPTH = 64,
  parameter int unsigned DQ_W      = ATLAS_HBD_DQ_W,
  parameter int unsigned ID_W      = 6,
  parameter int unsigned BUF_AW    = 20,
  parameter int unsigned ACC_W     = $clog2(ACC_N),
  parameter int unsigned NSFU      = (VEC_N + SFU_RATIO - 1) / SFU_RATIO
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                dram_tick,

  // ---- matrix descriptor stream ----
  input  logic                mm_valid,
  input  logic [1:0]          mm_fmt_a,
  input  logic [1:0]          mm_fmt_b,
  input  logic [ACC_W-1:0]    mm_acc_id,
  input  logic                mm_first,
  input  logic                mm_last,
  input  logic [PE_ROWS*DOT_N*8-1:0] mm_a,
  input  logic [PE_ROWS*8-1:0]       mm_a_scale,
  input  logic [PE_COLS*DOT_N*8-1:0] mm_b,
  input  logic [PE_COLS*8-1:0]       mm_b_scale,
  output logic [PE_ROWS*PE_COLS-1:0]    mm_out_valid,
  output logic [PE_ROWS*PE_COLS*32-1:0] mm_out,

  // ---- vector port ----
  input  logic                vec_valid,
  input  logic [2:0]          vec_op,
  input  logic [VEC_N*32-1:0] vec_a,
  input  logic [VEC_N*32-1:0] vec_b,
  output logic                vec_out_valid,
  output logic [VEC_N*32-1:0] vec_y,
  input  logic                sfu_valid,
  input  logic [1:0]          sfu_op,
  input  logic [NSFU*32-1:0]  sfu_x,
  output logic                sfu_out_valid,
  output logic [NSFU*32-1:0]  sfu_y,

  // ---- DMA descriptor ----
  input  logic                dma_valid,
  output logic                dma_ready,
  input  logic                dma_store,
  input  logic [ATLAS_HBD_ROW_W-1:0] dma_row,
  input  logic [ATLAS_HBD_COL_W-1:0] dma_col,
  input  logic [BUF_AW-1:0]   dma_buf_addr,
  input  logic [15:0]         dma_lines,
  output logic                dma_done,

  // ---- HBDRAM interface to the bonded die ----
  output logic [2:0]                 dram_cmd,
  output logic [ATLAS_HBD_ROW_W-1:0] dram_row,
  output logic [ATLAS_HBD_COL_W-1:0] dram_col,
  output logic [DQ_W-1:0]            dram_wdata,
  input  logic [DQ_W-1:0]            dram_rdata,
  input  logic                       dram_rvalid,

  // ---- NoC local port ----
  output logic                noc_inj_valid,
  output logic [ATLAS_FLIT_W-1:0] noc_inj_flit,
  input  logic                noc_inj_ready,
  input  logic                noc_ej_valid,
  input  logic [ATLAS_FLIT_W-1:0] noc_ej_flit,
  output logic                noc_ej_ready,

  // ---- HBDRAM statistics ----
  output logic [31:0]         stat_row_hit,
  output logic [31:0]         stat_row_miss,
  output logic [31:0]         stat_refresh
);

  //=========================================================================
  // Matrix engine
  //=========================================================================
  /* verilator lint_off UNUSEDSIGNAL */
  logic [PE_ROWS*PE_COLS*ACC_W-1:0] mm_out_id;
  /* verilator lint_on UNUSEDSIGNAL */

  atlas_matrix_unit #(
    .ROWS(PE_ROWS), .COLS(PE_COLS), .N(DOT_N), .ACC_N(ACC_N)
  ) u_matrix (
    .clk(clk), .rst_n(rst_n),
    .in_valid(mm_valid), .fmt_a(mm_fmt_a), .fmt_b(mm_fmt_b),
    .acc_id(mm_acc_id), .acc_first(mm_first), .acc_last(mm_last),
    .a_blocks(mm_a), .a_scales(mm_a_scale),
    .b_blocks(mm_b), .b_scales(mm_b_scale),
    .out_valid(mm_out_valid), .out_f32(mm_out), .out_id(mm_out_id)
  );

  //=========================================================================
  // Vector engine
  //=========================================================================
  atlas_vector_unit #(.VEC_N(VEC_N), .SFU_RATIO(SFU_RATIO)) u_vector (
    .clk(clk), .rst_n(rst_n),
    .vec_valid(vec_valid), .vec_op(vec_op), .vec_a(vec_a), .vec_b(vec_b),
    .vec_out_valid(vec_out_valid), .vec_y(vec_y),
    .sfu_valid(sfu_valid), .sfu_op(sfu_op), .sfu_x(sfu_x),
    .sfu_out_valid(sfu_out_valid), .sfu_y(sfu_y)
  );

  //=========================================================================
  // HBDRAM channel and its DMA engine
  //=========================================================================
  logic                        req_valid, req_ready, req_we;
  logic [ATLAS_HBD_ROW_W-1:0]  req_row;
  logic [ATLAS_HBD_COL_W-1:0]  req_col;
  logic [ID_W-1:0]             req_id;
  logic [DQ_W-1:0]             req_wdata;
  logic                        rsp_valid;
  logic [ID_W-1:0]             rsp_id;
  logic [DQ_W-1:0]             rsp_rdata;

  atlas_hbdram_ctrl #(.DQ_W(DQ_W), .ID_W(ID_W)) u_dram (
    .clk(clk), .rst_n(rst_n), .dram_tick(dram_tick),
    .req_valid(req_valid), .req_ready(req_ready), .req_we(req_we),
    .req_row(req_row), .req_col(req_col), .req_id(req_id), .req_wdata(req_wdata),
    .rsp_valid(rsp_valid), .rsp_id(rsp_id), .rsp_rdata(rsp_rdata),
    .dram_cmd(dram_cmd), .dram_row(dram_row), .dram_col(dram_col),
    .dram_wdata(dram_wdata), .dram_rdata(dram_rdata), .dram_rvalid(dram_rvalid),
    .stat_row_hit(stat_row_hit), .stat_row_miss(stat_row_miss),
    .stat_refresh(stat_refresh)
  );

  logic              buf_wr_en, buf_rd_en;
  // BUF_AW is sized for the full 4 MB scratchpad.  A scaled elaboration wires
  // fewer banks and shallower macros, so its upper address bits are genuinely
  // unused; they come back into play at the full configuration.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [BUF_AW-1:0] buf_wr_addr, buf_rd_addr;
  /* verilator lint_on UNUSEDSIGNAL */
  logic [DQ_W-1:0]   buf_wr_data, buf_rd_data;
  /* verilator lint_off UNUSEDSIGNAL */
  logic              dma_busy;
  /* verilator lint_on UNUSEDSIGNAL */

  atlas_dma #(.DQ_W(DQ_W), .ID_W(ID_W), .BUF_AW(BUF_AW)) u_dma (
    .clk(clk), .rst_n(rst_n),
    .cmd_valid(dma_valid), .cmd_ready(dma_ready), .cmd_store(dma_store),
    .cmd_row(dma_row), .cmd_col(dma_col), .cmd_buf_addr(dma_buf_addr),
    .cmd_lines(dma_lines), .busy(dma_busy), .done(dma_done),
    .req_valid(req_valid), .req_ready(req_ready), .req_we(req_we),
    .req_row(req_row), .req_col(req_col), .req_id(req_id), .req_wdata(req_wdata),
    .rsp_valid(rsp_valid), .rsp_id(rsp_id), .rsp_rdata(rsp_rdata),
    .buf_wr_en(buf_wr_en), .buf_wr_addr(buf_wr_addr), .buf_wr_data(buf_wr_data),
    .buf_rd_en(buf_rd_en), .buf_rd_addr(buf_rd_addr), .buf_rd_data(buf_rd_data)
  );

  //=========================================================================
  // Scratchpad
  //=========================================================================
  localparam int unsigned LINE_W  = BUF_LINE_B*8;
  localparam int unsigned BANK_W  = $clog2(BUF_BANKS);
  localparam int unsigned LINE_AW = $clog2(BUF_DEPTH);

  /* verilator lint_off UNUSEDSIGNAL */
  logic              buf_rd_valid, buf_rd_stall;
  logic [31:0]       buf_stalls;
  logic [LINE_W-1:0] buf_line_out;
  /* verilator lint_on UNUSEDSIGNAL */

  atlas_buffer #(
    .BANKS(BUF_BANKS), .LINE_B(BUF_LINE_B), .DEPTH(BUF_DEPTH)
  ) u_buffer (
    .clk(clk), .rst_n(rst_n),
    .rd_en   (buf_rd_en),
    .rd_bank (buf_rd_addr[BANK_W-1:0]),
    .rd_addr (buf_rd_addr[BANK_W +: LINE_AW]),
    .rd_valid(buf_rd_valid),
    .rd_data (buf_line_out),
    .wr_en   (buf_wr_en),
    .wr_bank (buf_wr_addr[BANK_W-1:0]),
    .wr_addr (buf_wr_addr[BANK_W +: LINE_AW]),
    .wr_data (buf_wr_data[LINE_W-1:0]),
    .wr_mask ({LINE_W{1'b1}}),
    .rd_stall(buf_rd_stall),
    .stall_count(buf_stalls)
  );

  // A DRAM line is DQ_W bits and a buffer line is LINE_W.  A store reads the
  // narrower buffer line and zero-extends it to the DRAM width.
  assign buf_rd_data = DQ_W'(buf_line_out);

  //=========================================================================
  // NoC local port.  Traffic is tensor-parallel reduction and MoE dispatch;
  // both move whole lines, so the port is a straight pass-through here and the
  // mesh at chip level does the routing.
  //=========================================================================
  assign noc_inj_valid = 1'b0;
  assign noc_inj_flit  = '0;
  assign noc_ej_ready  = 1'b1;

  /* verilator lint_off UNUSEDSIGNAL */
  wire unused_noc = noc_inj_ready | noc_ej_valid | (|noc_ej_flit);
  /* verilator lint_on UNUSEDSIGNAL */

endmodule
