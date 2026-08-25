//===========================================================================
// hbdram_model -- behavioural HBDRAM die, and a protocol checker
//
// Simulation only.  Beyond storing data, this enforces every timing
// constraint of the HBDRAM_500Mbps preset against the controller: it records
// when each command was issued and flags any command that arrives before the
// constraint it is subject to has expired.  A controller bug that would show
// up on silicon as a read of an unopened row, or an activate inside tRC, stops
// the simulation here instead.
//===========================================================================
`timescale 1ns/1ps
`include "atlas_defs.svh"

module hbdram_model #(
  parameter int unsigned DQ_W  = 1024,
  parameter int unsigned ROW_W = 13,
  parameter int unsigned COL_W = 9,
  // Only a small window of rows is backed by storage; a full 8192 x 512 x 1024
  // array is 4 Gbit and does not need to exist to test a controller.
  parameter int unsigned MODEL_ROWS = 16,
  parameter int unsigned MODEL_COLS = 32
) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             dram_tick,
  input  logic [2:0]       dram_cmd,
  input  logic [ROW_W-1:0] dram_row,
  input  logic [COL_W-1:0] dram_col,
  input  logic [DQ_W-1:0]  dram_wdata,
  output logic [DQ_W-1:0]  dram_rdata,
  output logic             dram_rvalid,
  output int               violations
);

  logic [DQ_W-1:0] mem [0:MODEL_ROWS*MODEL_COLS-1];

  logic             row_open;
  logic [ROW_W-1:0] open_row;

  // Cycle counters since each event, in DRAM clocks.
  int t_now;
  int t_act, t_pre, t_rd, t_wr, t_ref;

  function automatic int since(input int mark);
    since = (mark < 0) ? 32'sh7000_0000 : (t_now - mark);
  endfunction

  // Counting with a blocking assignment is deliberate: several checks can
  // fire on one command and each must be tallied within the same evaluation.
  /* verilator lint_off BLKSEQ */
  task automatic viol(input string what);
    violations = violations + 1;
    $display("%0t HBDRAM PROTOCOL VIOLATION: %s", $time, what);
  endtask
  /* verilator lint_on BLKSEQ */

  logic [DQ_W-1:0] rd_data_q;
  logic            rd_vld_q;

  initial begin
    violations = 0;
    t_now = 0; t_act = -1; t_pre = -1; t_rd = -1; t_wr = -1; t_ref = -1;
    row_open = 1'b0;
    rd_vld_q = 1'b0;
  end

  localparam int unsigned FA_W = $clog2(MODEL_ROWS*MODEL_COLS);
  wire [FA_W-1:0] flat_addr = FA_W'((32'(dram_row) % MODEL_ROWS) * MODEL_COLS
                                  + (32'(dram_col) % MODEL_COLS));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      row_open <= 1'b0;
      rd_vld_q <= 1'b0;
    end else if (dram_tick) begin
      t_now    <= t_now + 1;
      rd_vld_q <= 1'b0;

      case (dram_cmd)
        ATLAS_CMD_ACT: begin
          if (row_open)                            viol("ACT while a row is open");
          if (since(t_pre) < ATLAS_HBD_nRP)        viol("ACT violates tRP");
          if (since(t_act) < ATLAS_HBD_nRC)        viol("ACT violates tRC");
          if (since(t_ref) < ATLAS_HBD_nRFC)       viol("ACT violates tRFC");
          row_open <= 1'b1;
          open_row <= dram_row;
          t_act    <= t_now;
        end

        ATLAS_CMD_PRE: begin
          if (!row_open)                           viol("PRE with no row open");
          if (since(t_act) < ATLAS_HBD_nRAS)       viol("PRE violates tRAS");
          if (since(t_rd)  < ATLAS_HBD_nRTP)       viol("PRE violates tRTP");
          if (since(t_wr)  < ATLAS_HBD_nWR)        viol("PRE violates tWR");
          row_open <= 1'b0;
          t_pre    <= t_now;
        end

        ATLAS_CMD_RD: begin
          if (!row_open)                           viol("RD with no row open");
          else if (dram_row !== open_row)          viol("RD to a row that is not the open one");
          if (since(t_act) < ATLAS_HBD_nRCDRD)     viol("RD violates tRCD");
          if (since(t_wr)  < ATLAS_HBD_nWTR)       viol("RD violates tWTR");
          if (since(t_ref) < ATLAS_HBD_nRFC)       viol("RD violates tRFC");
          t_rd      <= t_now;
          rd_data_q <= mem[flat_addr];
          rd_vld_q  <= 1'b1;                       // nCL = 1
        end

        ATLAS_CMD_WR: begin
          if (!row_open)                           viol("WR with no row open");
          else if (dram_row !== open_row)          viol("WR to a row that is not the open one");
          if (since(t_act) < ATLAS_HBD_nRCDWR)     viol("WR violates tRCD");
          if (since(t_rd)  < ATLAS_HBD_nRTW)       viol("WR violates tRTW");
          if (since(t_ref) < ATLAS_HBD_nRFC)       viol("WR violates tRFC");
          t_wr           <= t_now;
          mem[flat_addr] <= dram_wdata;
        end

        ATLAS_CMD_REFAB: begin
          if (row_open)                            viol("REFab while a row is open");
          if (since(t_pre) < ATLAS_HBD_nRP)        viol("REFab violates tRP");
          t_ref <= t_now;
        end

        default: ;
      endcase
    end else begin
      rd_vld_q <= 1'b0;
    end
  end

  assign dram_rdata  = rd_data_q;
  assign dram_rvalid = rd_vld_q;

endmodule
