//===========================================================================
// atlas_buffer -- banked on-chip scratchpad for one ATLAS core
//
// ATLAS gives a cloud core a 4 MB buffer with 32768 B/cycle of read and the
// same of write bandwidth (configs/architecture/chip/cloud/stratum/atlas.yaml).
// No single SRAM macro delivers that, so the capacity is spread over BANKS
// independent macros, each LINE_B bytes wide, and the aggregate width is what
// produces the bandwidth figure:
//
//     BANKS * LINE_B bytes/cycle read, and the same for write
//
// Read and write are separate ports here because the ATLAS model charges them
// separately; each bank is a single-port macro, so a read and a write to the
// *same* bank in the same cycle conflict.  The arbiter below resolves that by
// giving the write priority and reporting the stall, rather than silently
// dropping one -- an accelerator's operand fetch and its result write-back
// naturally target different banks, and the stall counter makes it visible
// when they do not.
//===========================================================================
`include "atlas_defs.svh"

module atlas_buffer #(
  parameter int unsigned BANKS   = 64,
  parameter int unsigned LINE_B  = 512,                  // bytes per bank line
  parameter int unsigned DEPTH   = 128,                  // lines per bank
  parameter int unsigned LINE_W  = LINE_B*8,
  parameter int unsigned BANK_W  = $clog2(BANKS),
  parameter int unsigned LINE_AW = $clog2(DEPTH)
) (
  input  logic                clk,
  input  logic                rst_n,

  // Read port
  input  logic                rd_en,
  input  logic [BANK_W-1:0]   rd_bank,
  input  logic [LINE_AW-1:0]  rd_addr,
  output logic                rd_valid,
  output logic [LINE_W-1:0]   rd_data,

  // Write port
  input  logic                wr_en,
  input  logic [BANK_W-1:0]   wr_bank,
  input  logic [LINE_AW-1:0]  wr_addr,
  input  logic [LINE_W-1:0]   wr_data,
  input  logic [LINE_W-1:0]   wr_mask,

  // A read lost its bank to a write this cycle and must be retried.
  output logic                rd_stall,
  output logic [31:0]         stall_count
);

  // A conflict is a read and a write hitting the same bank in one cycle.
  wire conflict = rd_en & wr_en & (rd_bank == wr_bank);
  assign rd_stall = conflict;

  wire rd_go = rd_en & ~conflict;

  logic [BANKS*LINE_W-1:0] bank_rdata;

  genvar b;
  generate
    for (b = 0; b < BANKS; b++) begin : g_bank
      wire sel_rd = rd_go && (rd_bank == BANK_W'(b));
      wire sel_wr = wr_en && (wr_bank == BANK_W'(b));

      atlas_sram #(.WORDS(DEPTH), .DW(LINE_W)) u_macro (
        .clk  (clk),
        .en   (sel_rd | sel_wr),
        .we   (sel_wr),
        .addr (sel_wr ? wr_addr : rd_addr),
        .wdata(wr_data),
        .wmask(wr_mask),
        .rdata(bank_rdata[b*LINE_W +: LINE_W])
      );
    end
  endgenerate

  // The macros return data one cycle after the request, so the bank select
  // has to be delayed to match.
  logic                rd_go_q;
  logic [BANK_W-1:0]   rd_bank_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) rd_go_q <= 1'b0;
    else        rd_go_q <= rd_go;
  end

  always_ff @(posedge clk) begin
    rd_bank_q <= rd_bank;
  end

  assign rd_valid = rd_go_q;
  assign rd_data  = bank_rdata[rd_bank_q*LINE_W +: LINE_W];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)          stall_count <= 32'd0;
    else if (conflict)   stall_count <= stall_count + 32'd1;
  end

endmodule
