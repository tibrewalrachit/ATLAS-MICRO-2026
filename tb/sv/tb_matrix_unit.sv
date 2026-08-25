//===========================================================================
// tb_matrix_unit -- GEMM test of the systolic PE array.
//
// Drives one output-stationary pass and checks every PE's accumulated result
// against tb/golden/gen_gemm_vectors.py.  This is the test that proves the
// internal operand skew is right: if the row and column delays did not cancel,
// PEs would pair mismatched K blocks and only the (r == c) diagonal would
// still agree with the reference.
//===========================================================================
`timescale 1ns/1ps

module tb_matrix_unit;

  // Overridable so one testbench can sweep array geometries; the vector file
  // carries the geometry it was generated for and the header check below
  // refuses a mismatch.
  parameter int unsigned ROWS  = 3;
  parameter int unsigned COLS  = 4;
  parameter int unsigned N     = 32;
  parameter int unsigned ACC_N = 4;
  localparam int unsigned ACC_W = $clog2(ACC_N);
  localparam int unsigned BLK_W = N*8;
  localparam int unsigned MAXC  = 256;

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  logic                    in_valid;
  logic [1:0]              fmt_a, fmt_b;
  logic [ACC_W-1:0]        acc_id;
  logic                    acc_first, acc_last;
  logic [ROWS*BLK_W-1:0]   a_blocks;
  logic [ROWS*8-1:0]       a_scales;
  logic [COLS*BLK_W-1:0]   b_blocks;
  logic [COLS*8-1:0]       b_scales;
  logic [ROWS*COLS-1:0]        out_valid;
  logic [ROWS*COLS*32-1:0]     out_f32;
  logic [ROWS*COLS*ACC_W-1:0]  out_id;

  atlas_matrix_unit #(.ROWS(ROWS), .COLS(COLS), .N(N), .ACC_N(ACC_N)) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .fmt_a(fmt_a), .fmt_b(fmt_b),
    .acc_id(acc_id), .acc_first(acc_first), .acc_last(acc_last),
    .a_blocks(a_blocks), .a_scales(a_scales),
    .b_blocks(b_blocks), .b_scales(b_scales),
    .out_valid(out_valid), .out_f32(out_f32), .out_id(out_id)
  );

  // Stimulus storage
  logic [ACC_W-1:0]      c_id   [0:MAXC-1];
  logic                  c_fst  [0:MAXC-1];
  logic                  c_lst  [0:MAXC-1];
  logic [ROWS*BLK_W-1:0] c_a    [0:MAXC-1];
  logic [ROWS*8-1:0]     c_as   [0:MAXC-1];
  logic [COLS*BLK_W-1:0] c_b    [0:MAXC-1];
  logic [COLS*8-1:0]     c_bs   [0:MAXC-1];

  // Expected results, indexed [row][col][tile]
  logic [31:0] exp_v [0:ROWS*COLS*ACC_N-1];
  logic        got_f [0:ROWS*COLS*ACC_N-1];

  int ncyc = 0, nexp = 0, errors = 0, results = 0;

  // ---------------- result collection ----------------
  genvar gr, gc;
  generate
    for (gr = 0; gr < ROWS; gr++) begin : g_chk_r
      for (gc = 0; gc < COLS; gc++) begin : g_chk_c
        localparam int unsigned P = gr*COLS + gc;
        always @(posedge clk) begin
          if (rst_n && out_valid[P]) begin
            automatic int tile = int'(out_id[P*ACC_W +: ACC_W]);
            automatic int idx  = (gr*COLS + gc)*ACC_N + tile;
            results = results + 1;
            if (got_f[idx]) begin
              errors = errors + 1;
              $display("  DUPLICATE result for r=%0d c=%0d tile=%0d", gr, gc,
                       out_id[P*ACC_W +: ACC_W]);
            end
            got_f[idx] = 1'b1;
            if (out_f32[P*32 +: 32] !== exp_v[idx]) begin
              errors = errors + 1;
              if (errors < 10)
                $display("  MISMATCH r=%0d c=%0d tile=%0d  got=%08x exp=%08x",
                         gr, gc, out_id[P*ACC_W +: ACC_W],
                         out_f32[P*32 +: 32], exp_v[idx]);
            end
          end
        end
      end
    end
  endgenerate

  // ---------------- stimulus ----------------
  int    fd, code, i, r, c, t, rr, cc, tt;
  int    f_rows, f_cols, f_k, f_accn, f_fa, f_fb;
  string tag, fname;
  logic [31:0]      ev;
  logic [BLK_W-1:0] tmp_blk;
  logic [7:0]       tmp_sc;

  initial begin
    in_valid = 1'b0; acc_id = '0; acc_first = 1'b0; acc_last = 1'b0;
    a_blocks = '0; a_scales = '0; b_blocks = '0; b_scales = '0;
    for (i = 0; i < ROWS*COLS*ACC_N; i = i + 1) got_f[i] = 1'b0;

    if (!$value$plusargs("vectors=%s", fname)) fname = "tb/sv/gemm_vectors.txt";
    fd = $fopen(fname, "r");
    if (fd == 0) begin $display("TB ERROR: cannot open %s", fname); $fatal(1); end

    code = $fscanf(fd, "%s %d %d %d %d %d %d", tag,
                   f_rows, f_cols, f_k, f_accn, f_fa, f_fb);
    if (code != 7 || tag != "HEADER") begin
      $display("TB ERROR: bad header (code=%0d tag=%s)", code, tag);
      $fatal(1);
    end
    if (f_rows != ROWS || f_cols != COLS || f_accn != ACC_N) begin
      $display("TB ERROR: vector geometry %0dx%0d/%0d does not match DUT %0dx%0d/%0d",
               f_rows, f_cols, f_accn, ROWS, COLS, ACC_N);
      $fatal(1);
    end
    fmt_a = f_fa[1:0];
    fmt_b = f_fb[1:0];

    // Read tagged records until end of file.
    code = 1;
    while (code == 1) begin
      code = $fscanf(fd, "%s", tag);
      if (code == 1) begin
        if (tag == "CYCLE") begin
          void'($fscanf(fd, "%d %d %d", t, rr, cc));
          c_id[ncyc]  = t[ACC_W-1:0];
          c_fst[ncyc] = rr[0];
          c_lst[ncyc] = cc[0];
          // $fscanf needs a plain lvalue, not a part-select of an array
          // element, so each field is read into a temporary first.
          for (i = 0; i < ROWS; i = i + 1) begin
            void'($fscanf(fd, "%h", tmp_blk));
            c_a[ncyc][i*BLK_W +: BLK_W] = tmp_blk;
          end
          for (i = 0; i < ROWS; i = i + 1) begin
            void'($fscanf(fd, "%h", tmp_sc));
            c_as[ncyc][i*8 +: 8] = tmp_sc;
          end
          for (i = 0; i < COLS; i = i + 1) begin
            void'($fscanf(fd, "%h", tmp_blk));
            c_b[ncyc][i*BLK_W +: BLK_W] = tmp_blk;
          end
          for (i = 0; i < COLS; i = i + 1) begin
            void'($fscanf(fd, "%h", tmp_sc));
            c_bs[ncyc][i*8 +: 8] = tmp_sc;
          end
          ncyc = ncyc + 1;
        end else if (tag == "EXPECT") begin
          void'($fscanf(fd, "%d %d %d %h", r, c, tt, ev));
          exp_v[(r*COLS + c)*ACC_N + tt] = ev;
          nexp = nexp + 1;
        end
      end
    end
    $fclose(fd);
    $display("tb_matrix_unit: %0d issue cycles, %0d expected results", ncyc, nexp);

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    for (i = 0; i < ncyc; i = i + 1) begin
      @(negedge clk);
      in_valid  = 1'b1;
      acc_id    = c_id[i];
      acc_first = c_fst[i];
      acc_last  = c_lst[i];
      a_blocks  = c_a[i];
      a_scales  = c_as[i];
      b_blocks  = c_b[i];
      b_scales  = c_bs[i];
    end
    @(negedge clk);
    in_valid = 1'b0; acc_first = 1'b0; acc_last = 1'b0;

    // Drain: edge skew (ROWS-1 + COLS-1) + array walk + PE pipeline.
    repeat (ROWS + COLS + 40) @(negedge clk);

    if (results != nexp) begin
      $display("  result count %0d != expected %0d", results, nexp);
      errors = errors + 1;
    end
    if (errors == 0)
      $display("TB_MATRIX_UNIT: PASS  (%0dx%0d array, %0d results, 0 mismatches)",
               ROWS, COLS, results);
    else
      $display("TB_MATRIX_UNIT: FAIL  (%0d errors, %0d/%0d results)",
               errors, results, nexp);
    $finish;
  end

endmodule
