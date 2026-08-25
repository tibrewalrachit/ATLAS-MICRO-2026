//===========================================================================
// atlas_hbdram_ctrl -- one channel of the ATLAS hybrid-bonding DRAM
//
// HBDRAM is the memory ATLAS is built around: a DRAM die bonded face to face
// with the logic die rather than attached through a package.  The organisation
// and timing here are the ones the ATLAS simulator uses -- preset
// HBDRAM_4Gb_1024pin_512col with timing HBDRAM_500Mbps, from the ramulator2
// patch in patches/ramulator2.patch:
//
//     1024 DQ per channel, 8192 rows, 512 columns, one bank
//     tCK 2 ns (500 MT/s), nBL 1, nCL 1, nRCD 8, nRP 6, nRAS 17, nRC 23
//
// 1024 bits per transfer at 500 MT/s is 64 GB/s per channel, and the 16
// channels of configs/architecture/dram/cloud/cloud_1TBps.yaml make 1 TB/s.
//
// Two things about that organisation are worth stating, because they are what
// hybrid bonding buys and they shape this controller:
//
//   * nCL and nBL are 1.  There is no off-package SerDes to cross, so a column
//     read returns its 128 bytes essentially immediately.  Read latency is
//     dominated by row activation, not by the data path.
//   * There is one bank per channel, not the eight or sixteen of an HBM stack.
//     Bonding pitch buys channel parallelism instead of bank parallelism, so
//     there is no bank-level interleaving to hide activation behind -- row
//     locality within a channel is the only lever the scheduler has, which is
//     exactly why the policy below is FR-FCFS over an open row.
//
// The controller runs on the DRAM clock, which is half the core clock.  Both
// come from the same PLL, so instead of an asynchronous crossing the whole
// block is clocked at the core rate and advances only on `dram_tick`.  That
// keeps one clock domain in the core while still charging every DRAM timing
// constraint in real tCK units.
//===========================================================================
`include "atlas_defs.svh"

module atlas_hbdram_ctrl #(
  parameter int unsigned DQ_W    = ATLAS_HBD_DQ_W,     // 1024
  parameter int unsigned ROW_W   = ATLAS_HBD_ROW_W,    // 13
  parameter int unsigned COL_W   = ATLAS_HBD_COL_W,    // 9
  parameter int unsigned QDEPTH  = 16,
  parameter int unsigned ID_W    = 6,
  parameter int unsigned QPTR_W  = $clog2(QDEPTH)
) (
  input  logic                clk,
  input  logic                rst_n,
  // One pulse per DRAM clock.  The controller's timing counters advance only
  // on these, so every constraint below is expressed in real tCK.
  input  logic                dram_tick,

  // ---- request port (core clock) ----
  input  logic                req_valid,
  output logic                req_ready,
  input  logic                req_we,
  input  logic [ROW_W-1:0]    req_row,
  input  logic [COL_W-1:0]    req_col,
  input  logic [ID_W-1:0]     req_id,
  input  logic [DQ_W-1:0]     req_wdata,

  // ---- response port ----
  output logic                rsp_valid,
  output logic [ID_W-1:0]     rsp_id,
  output logic [DQ_W-1:0]     rsp_rdata,

  // ---- DRAM command interface (to the bonded die) ----
  output logic [2:0]          dram_cmd,      // ATLAS_CMD_*
  output logic [ROW_W-1:0]    dram_row,
  output logic [COL_W-1:0]    dram_col,
  output logic [DQ_W-1:0]     dram_wdata,
  input  logic [DQ_W-1:0]     dram_rdata,
  input  logic                dram_rvalid,

  // ---- statistics ----
  output logic [31:0]         stat_row_hit,
  output logic [31:0]         stat_row_miss,
  output logic [31:0]         stat_refresh
);

  //=========================================================================
  // Request queue
  //=========================================================================
  logic               q_vld  [0:QDEPTH-1];
  logic               q_we   [0:QDEPTH-1];
  logic [ROW_W-1:0]   q_row  [0:QDEPTH-1];
  logic [COL_W-1:0]   q_col  [0:QDEPTH-1];
  logic [ID_W-1:0]    q_id   [0:QDEPTH-1];
  logic [DQ_W-1:0]    q_data [0:QDEPTH-1];
  // Arrival order, stamped from a free-running counter.  A plain age counter
  // incremented per DRAM tick cannot order two requests accepted between the
  // same pair of ticks, and the core issues faster than the DRAM clock.
  logic [SEQ_W-1:0]   q_seq  [0:QDEPTH-1];
  logic [SEQ_W-1:0]   seq_ctr;
  // Rebasing reference for the arbitration trees.  It only has to be no newer
  // than every queued entry; the counter value from when the queue last
  // emptied satisfies that, and the queue holds at most QDEPTH entries out of
  // a 2^SEQ_W space, so the rebased keys never wrap.
  logic [SEQ_W-1:0]   base_seq;

  localparam int unsigned SEQ_W = 16;

  logic [QPTR_W-1:0]  free_slot;
  logic               has_free;

  integer fi;
  always_comb begin
    has_free  = 1'b0;
    free_slot = '0;
    for (fi = QDEPTH-1; fi >= 0; fi = fi - 1) begin
      if (!q_vld[fi]) begin
        has_free  = 1'b1;
        free_slot = QPTR_W'(fi);
      end
    end
  end

  assign req_ready = has_free;

  //=========================================================================
  // Bank state and timing counters (all in DRAM clocks)
  //=========================================================================
  localparam int unsigned CW = 12;   // wide enough for nRFC and nREFI

  logic             row_open;
  logic [ROW_W-1:0] open_row;

  logic [CW-1:0] c_rcd;    // ACT -> RD/WR
  logic [CW-1:0] c_ras;    // ACT -> PRE
  logic [CW-1:0] c_rc;     // ACT -> ACT
  logic [CW-1:0] c_rp;     // PRE -> ACT
  logic [CW-1:0] c_rtp;    // RD  -> PRE
  logic [CW-1:0] c_wr;     // WR  -> PRE
  logic [CW-1:0] c_wtr;    // WR  -> RD
  logic [CW-1:0] c_rtw;    // RD  -> WR
  logic [CW-1:0] c_rfc;    // refresh busy
  logic [CW-1:0] c_refi;   // time to next refresh

  wire can_act = (c_rp == '0) && (c_rc == '0) && (c_rfc == '0);
  wire can_rd  = row_open && (c_rcd == '0) && (c_wtr == '0) && (c_rfc == '0);
  wire can_wr  = row_open && (c_rcd == '0) && (c_rtw == '0) && (c_rfc == '0);
  wire can_pre = row_open && (c_ras == '0) && (c_rtp == '0) && (c_wr == '0);
  // A refresh needs the array closed, and it takes priority once due.
  wire ref_due = (c_refi == '0);
  wire can_ref = !row_open && (c_rp == '0) && (c_rfc == '0);

  //=========================================================================
  // FR-FCFS selection
  //
  // First pass: the oldest request that hits the open row and is ready to
  // issue.  Second pass: the oldest request of any kind, which decides which
  // row to open next.  With a single bank per channel there is nothing else to
  // overlap an activation with, so keeping a hot row open is the whole game.
  //=========================================================================
  logic              hit_found, any_found;
  logic [QPTR_W-1:0] hit_slot,  any_slot;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [SEQ_W-1:0]  hit_seq,   any_seq;
  /* verilator lint_on UNUSEDSIGNAL */

  // A request may not overtake an older request to the *same address*.
  // Without this, FR-FCFS will happily promote a read that hits the open row
  // ahead of a queued write to the very same location, and the requester sees
  // stale data -- reordering is only safe between independent addresses.
  logic [QDEPTH-1:0] blocked;

  integer bi, bj;
  always_comb begin
    for (bi = 0; bi < QDEPTH; bi = bi + 1) begin
      blocked[bi] = 1'b0;
      for (bj = 0; bj < QDEPTH; bj = bj + 1) begin
        if ((bi != bj) && q_vld[bi] && q_vld[bj] &&
            (q_row[bi] == q_row[bj]) && (q_col[bi] == q_col[bj]) &&
            ($signed(q_seq[bj] - q_seq[bi]) < 0))
          blocked[bi] = 1'b1;
      end
    end
  end

  // Selection uses tournament trees rather than a scan over the queue.  The
  // scan form chained QDEPTH full sequence-number comparisons and was the
  // worst path in the whole design by a wide margin.
  //
  // Sequence numbers are rebased against the oldest entry before comparison,
  // so the tree's unsigned compare stays monotonic across a counter wrap.
  logic [QDEPTH-1:0]       cand_any, cand_hit;
  logic [QDEPTH*SEQ_W-1:0] rel_key;

  logic [QDEPTH-1:0] q_vld_v;

  integer ci;
  always_comb begin
    for (ci = 0; ci < QDEPTH; ci = ci + 1) begin
      q_vld_v[ci]  = q_vld[ci];
      cand_any[ci] = q_vld[ci] & ~blocked[ci];
      cand_hit[ci] = q_vld[ci] & ~blocked[ci] & row_open &&
                     (q_row[ci] == open_row) &&
                     (q_we[ci] ? can_wr : can_rd);
      rel_key[ci*SEQ_W +: SEQ_W] = q_seq[ci] - base_seq;
    end
  end

  atlas_arb_tree #(.N(QDEPTH), .KW(SEQ_W)) u_arb_any (
    .valid(cand_any), .key(rel_key),
    .out_valid(any_found), .out_idx(any_slot), .out_key(any_seq)
  );

  atlas_arb_tree #(.N(QDEPTH), .KW(SEQ_W)) u_arb_hit (
    .valid(cand_hit), .key(rel_key),
    .out_valid(hit_found), .out_idx(hit_slot), .out_key(hit_seq)
  );

  // A miss must precharge before the wanted row can be activated.
  wire need_pre = any_found && row_open && (q_row[any_slot] != open_row);
  wire need_act = any_found && !row_open;

  //=========================================================================
  // Command issue.  One DRAM command per dram_tick, at most.
  //=========================================================================
  logic [2:0]        issue_cmd;
  logic [QPTR_W-1:0] issue_slot;

  always_comb begin
    issue_cmd  = ATLAS_CMD_NOP;
    issue_slot = '0;

    if (ref_due && can_ref) begin
      issue_cmd = ATLAS_CMD_REFAB;
    end else if (ref_due && row_open && can_pre) begin
      // Close the row so the refresh can proceed.
      issue_cmd = ATLAS_CMD_PRE;
    end else if (hit_found) begin
      issue_cmd  = q_we[hit_slot] ? ATLAS_CMD_WR : ATLAS_CMD_RD;
      issue_slot = hit_slot;
    end else if (need_pre && can_pre) begin
      issue_cmd = ATLAS_CMD_PRE;
    end else if (need_act && can_act) begin
      issue_cmd  = ATLAS_CMD_ACT;
      issue_slot = any_slot;
    end
  end

  wire do_issue = dram_tick && (issue_cmd != ATLAS_CMD_NOP);

  assign dram_cmd   = do_issue ? issue_cmd : ATLAS_CMD_NOP;
  assign dram_row   = (issue_cmd == ATLAS_CMD_ACT) ? q_row[any_slot] : open_row;
  assign dram_col   = q_col[issue_slot];
  assign dram_wdata = q_data[issue_slot];

  //=========================================================================
  // Sequential state
  //=========================================================================
  integer k;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (k = 0; k < QDEPTH; k = k + 1) q_vld[k] <= 1'b0;
      seq_ctr  <= '0;
      base_seq <= '0;
      row_open      <= 1'b0;
      open_row      <= '0;
      c_rcd <= '0; c_ras <= '0; c_rc  <= '0; c_rp  <= '0; c_rtp <= '0;
      c_wr  <= '0; c_wtr <= '0; c_rtw <= '0; c_rfc <= '0;
      c_refi        <= CW'(ATLAS_HBD_nREFI);
      stat_row_hit  <= 32'd0;
      stat_row_miss <= 32'd0;
      stat_refresh  <= 32'd0;
    end else begin
      // The queue being empty is the moment the rebasing reference can safely
      // advance: nothing older than seq_ctr is outstanding.
      if (q_vld_v == '0) base_seq <= seq_ctr;

      // ---- accept a new request ----
      if (req_valid && has_free) begin
        q_vld [free_slot] <= 1'b1;
        q_we  [free_slot] <= req_we;
        q_row [free_slot] <= req_row;
        q_col [free_slot] <= req_col;
        q_id  [free_slot] <= req_id;
        q_data[free_slot] <= req_wdata;
        q_seq [free_slot] <= seq_ctr;
        seq_ctr           <= seq_ctr + 1'b1;
      end

      // ---- timing counters advance on the DRAM clock only ----
      if (dram_tick) begin
        if (c_rcd != '0) c_rcd <= c_rcd - 1'b1;
        if (c_ras != '0) c_ras <= c_ras - 1'b1;
        if (c_rc  != '0) c_rc  <= c_rc  - 1'b1;
        if (c_rp  != '0) c_rp  <= c_rp  - 1'b1;
        if (c_rtp != '0) c_rtp <= c_rtp - 1'b1;
        if (c_wr  != '0) c_wr  <= c_wr  - 1'b1;
        if (c_wtr != '0) c_wtr <= c_wtr - 1'b1;
        if (c_rtw != '0) c_rtw <= c_rtw - 1'b1;
        if (c_rfc != '0) c_rfc <= c_rfc - 1'b1;
        if (c_refi != '0) c_refi <= c_refi - 1'b1;
      end

      // ---- apply the issued command ----
      if (do_issue) begin
        case (issue_cmd)
          ATLAS_CMD_ACT: begin
            row_open <= 1'b1;
            open_row <= q_row[any_slot];
            c_rcd    <= CW'(ATLAS_HBD_nRCDRD);
            c_ras    <= CW'(ATLAS_HBD_nRAS);
            c_rc     <= CW'(ATLAS_HBD_nRC);
            stat_row_miss <= stat_row_miss + 1'b1;
          end
          ATLAS_CMD_PRE: begin
            row_open <= 1'b0;
            c_rp     <= CW'(ATLAS_HBD_nRP);
          end
          ATLAS_CMD_RD: begin
            q_vld[hit_slot] <= 1'b0;
            c_rtp <= CW'(ATLAS_HBD_nRTP);
            c_rtw <= CW'(ATLAS_HBD_nRTW);
            stat_row_hit <= stat_row_hit + 1'b1;
          end
          ATLAS_CMD_WR: begin
            q_vld[hit_slot] <= 1'b0;
            c_wr  <= CW'(ATLAS_HBD_nWR);
            c_wtr <= CW'(ATLAS_HBD_nWTR);
            stat_row_hit <= stat_row_hit + 1'b1;
          end
          ATLAS_CMD_REFAB: begin
            c_rfc  <= CW'(ATLAS_HBD_nRFC);
            c_refi <= CW'(ATLAS_HBD_nREFI);
            stat_refresh <= stat_refresh + 1'b1;
          end
          default: ;
        endcase
      end
    end
  end

  //=========================================================================
  // Read response.  nCL is 1, so the bonded die returns data almost at once;
  // the id is carried alongside so the core can match out-of-order returns,
  // which FR-FCFS will produce whenever it favours a row hit over an older
  // request.
  //=========================================================================
  // Reads can issue on consecutive DRAM ticks, so the ids in flight are held
  // in a small FIFO.  A single register would be overwritten by the second
  // read before the first one's data came back, and the response would carry
  // the wrong id.
  localparam int unsigned RDF_DEPTH = 4;
  localparam int unsigned RDF_PTR_W = $clog2(RDF_DEPTH);

  logic [ID_W-1:0]     rdf     [0:RDF_DEPTH-1];
  logic [RDF_PTR_W:0]  rdf_wr, rdf_rd;

  wire rdf_empty = (rdf_wr == rdf_rd);
  wire rd_issued = do_issue && (issue_cmd == ATLAS_CMD_RD);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rdf_wr <= '0;
      rdf_rd <= '0;
    end else begin
      if (rd_issued) begin
        rdf[rdf_wr[RDF_PTR_W-1:0]] <= q_id[hit_slot];
        rdf_wr <= rdf_wr + 1'b1;
      end
      if (dram_rvalid && !rdf_empty) rdf_rd <= rdf_rd + 1'b1;
    end
  end

  assign rsp_valid = dram_rvalid & ~rdf_empty;
  assign rsp_id    = rdf[rdf_rd[RDF_PTR_W-1:0]];
  assign rsp_rdata = dram_rdata;

endmodule
