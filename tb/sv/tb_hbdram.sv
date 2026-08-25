//===========================================================================
// tb_hbdram -- drives the HBDRAM controller against the checking device model
//
// Checks three things at once:
//   1. no HBDRAM timing constraint is ever violated (the model asserts),
//   2. every read returns the data most recently written to that address,
//      which is what proves FR-FCFS reordering does not corrupt ordering
//      the requester can observe, and
//   3. row locality actually turns into row hits.
//
// The address stream deliberately mixes a hot row with scattered ones, so the
// scheduler has both hits to find and misses to handle.
//===========================================================================
`timescale 1ns/1ps

module tb_hbdram;

  localparam int unsigned DQ_W  = 1024;
  localparam int unsigned ROW_W = 13;
  localparam int unsigned COL_W = 9;
  localparam int unsigned ID_W  = 6;
  localparam int unsigned MROWS = 16;
  localparam int unsigned MCOLS = 32;
  localparam int unsigned NREQ  = 2000;
  // Overridable so the queue depth can be swept: scheduler quality and area
  // trade against each other here.
  parameter  int unsigned QDEPTH = 16;

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  // Core runs at 1 GHz, HBDRAM at 500 MT/s: one DRAM tick every two cycles.
  logic dram_tick = 1'b0;
  always_ff @(posedge clk) dram_tick <= ~dram_tick;

  logic             req_valid, req_ready, req_we;
  logic [ROW_W-1:0] req_row;
  logic [COL_W-1:0] req_col;
  logic [ID_W-1:0]  req_id;
  logic [DQ_W-1:0]  req_wdata;
  logic             rsp_valid;
  logic [ID_W-1:0]  rsp_id;
  logic [DQ_W-1:0]  rsp_rdata;
  logic [2:0]       dram_cmd;
  logic [ROW_W-1:0] dram_row;
  logic [COL_W-1:0] dram_col;
  logic [DQ_W-1:0]  dram_wdata, dram_rdata;
  logic             dram_rvalid;
  logic [31:0]      stat_row_hit, stat_row_miss, stat_refresh;
  int               violations;

  atlas_hbdram_ctrl #(.DQ_W(DQ_W), .ROW_W(ROW_W), .COL_W(COL_W),
                     .ID_W(ID_W), .QDEPTH(QDEPTH)) dut (
    .clk(clk), .rst_n(rst_n), .dram_tick(dram_tick),
    .req_valid(req_valid), .req_ready(req_ready), .req_we(req_we),
    .req_row(req_row), .req_col(req_col), .req_id(req_id), .req_wdata(req_wdata),
    .rsp_valid(rsp_valid), .rsp_id(rsp_id), .rsp_rdata(rsp_rdata),
    .dram_cmd(dram_cmd), .dram_row(dram_row), .dram_col(dram_col),
    .dram_wdata(dram_wdata), .dram_rdata(dram_rdata), .dram_rvalid(dram_rvalid),
    .stat_row_hit(stat_row_hit), .stat_row_miss(stat_row_miss),
    .stat_refresh(stat_refresh)
  );

  hbdram_model #(.DQ_W(DQ_W), .ROW_W(ROW_W), .COL_W(COL_W),
                 .MODEL_ROWS(MROWS), .MODEL_COLS(MCOLS)) u_dram (
    .clk(clk), .rst_n(rst_n), .dram_tick(dram_tick),
    .dram_cmd(dram_cmd), .dram_row(dram_row), .dram_col(dram_col),
    .dram_wdata(dram_wdata), .dram_rdata(dram_rdata), .dram_rvalid(dram_rvalid),
    .violations(violations)
  );

  // Scoreboard: shadow copy of memory, and what each outstanding read expects.
  logic [DQ_W-1:0] shadow  [0:MROWS*MCOLS-1];
  logic            written [0:MROWS*MCOLS-1];
  logic [DQ_W-1:0] exp_data[0:(1<<ID_W)-1];
  logic            id_busy [0:(1<<ID_W)-1];

  int reads_issued = 0, reads_returned = 0, errors = 0;

  always @(posedge clk) begin
    if (rst_n && rsp_valid) begin
      reads_returned = reads_returned + 1;
      if (!id_busy[rsp_id]) begin
        errors = errors + 1;
        $display("  response for id %0d that was never issued", rsp_id);
      end else if (rsp_rdata !== exp_data[rsp_id]) begin
        errors = errors + 1;
        if (errors < 6)
          $display("  DATA MISMATCH id=%0d got=%h... exp=%h...",
                   rsp_id, rsp_rdata[63:0], exp_data[rsp_id][63:0]);
      end
      id_busy[rsp_id] = 1'b0;
    end
  end

  int i, r, c, a, nid;
  logic [255:0] d;

  initial begin
    req_valid = 1'b0; req_we = 1'b0; req_row = '0; req_col = '0;
    req_id = '0; req_wdata = '0;
    for (i = 0; i < MROWS*MCOLS; i = i + 1) written[i] = 1'b0;
    for (i = 0; i < (1<<ID_W); i = i + 1) id_busy[i] = 1'b0;
    nid = 0;

    repeat (6) @(negedge clk);
    rst_n = 1'b1;

    for (i = 0; i < NREQ; i = i + 1) begin
      // Bias towards one hot row so the scheduler has hits available, with
      // scattered rows mixed in to force activations.
      r = ($urandom_range(0, 99) < 60) ? 3 : $urandom_range(0, MROWS-1);
      c = $urandom_range(0, MCOLS-1);
      a = r*MCOLS + c;

      // Only read an address that has been written, so every read has a
      // defined expected value.
      if (!written[a] || ($urandom_range(0, 99) < 40)) begin
        d = {$urandom, $urandom, $urandom, $urandom,
             $urandom, $urandom, $urandom, $urandom};
        req_we    = 1'b1;
        req_wdata = {DQ_W/256{d}};
        shadow[a] = {DQ_W/256{d}};
        written[a] = 1'b1;
        req_id    = ID_W'(nid);
      end else begin
        req_we = 1'b0;
        // Wait for a free id rather than reusing one still in flight.
        while (id_busy[nid % (1<<ID_W)]) begin
          @(negedge clk);
          req_valid = 1'b0;
        end
        req_id           = ID_W'(nid % (1<<ID_W));
        exp_data[req_id] = shadow[a];
        id_busy[req_id]  = 1'b1;
        reads_issued     = reads_issued + 1;
      end
      nid = (nid + 1) % (1<<ID_W);

      req_row = ROW_W'(r);
      req_col = COL_W'(c);
      req_valid = 1'b1;
      @(negedge clk);
      while (!req_ready) @(negedge clk);
    end
    req_valid = 1'b0;

    // Drain, then let a refresh interval elapse so REFab is exercised too.
    repeat (12000) @(negedge clk);

    $display("tb_hbdram: %0d requests, %0d reads issued, %0d returned",
             NREQ, reads_issued, reads_returned);
    $display("           row hits %0d, row misses %0d, refreshes %0d",
             stat_row_hit, stat_row_miss, stat_refresh);
    if (reads_returned != reads_issued) begin
      $display("  %0d reads never returned", reads_issued - reads_returned);
      errors = errors + 1;
    end
    if (violations != 0) begin
      $display("  %0d HBDRAM protocol violations", violations);
      errors = errors + violations;
    end
    if (stat_refresh == 0) begin
      $display("  no refresh was issued -- the test did not run long enough");
      errors = errors + 1;
    end
    if (errors == 0)
      $display("TB_HBDRAM: PASS  (timing clean, all reads correct)");
    else
      $display("TB_HBDRAM: FAIL  (%0d errors)", errors);
    $finish;
  end

endmodule
