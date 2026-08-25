//===========================================================================
// tb_rmsnorm -- checks the two-pass RMS normalisation against a reference
//
// Streams a row through the accumulate pass, waits for the scale, then streams
// it again through the scale pass and compares every output.
//===========================================================================
`timescale 1ns/1ps

module tb_rmsnorm;

  localparam int unsigned NROW = 64;    // elements per row
  localparam int unsigned NTEST = 16;

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  logic        start, acc_valid, acc_last, scale_ready;
  logic [31:0] inv_n, eps, acc_x, scale;
  logic        mul_valid, out_valid;
  logic [31:0] mul_x, mul_w, out_y;

  atlas_rmsnorm dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .inv_n(inv_n), .eps(eps),
    .acc_valid(acc_valid), .acc_x(acc_x), .acc_last(acc_last),
    .scale_ready(scale_ready), .scale(scale),
    .mul_valid(mul_valid), .mul_x(mul_x), .mul_w(mul_w),
    .out_valid(out_valid), .out_y(out_y)
  );

  logic [31:0] xs [0:NROW-1];
  logic [31:0] ws [0:NROW-1];
  logic [31:0] ex [0:NROW-1];
  int got = 0, errors = 0, t, i;

  function automatic real f32(input logic [31:0] b);
    real m; int e, k;
    if (b[30:23] == 8'd0) f32 = 0.0;
    else begin
      m = 1.0;
      for (k = 0; k < 23; k = k + 1) if (b[k]) m = m + (2.0 ** (k - 23));
      e = int'(b[30:23]) - 127;
      f32 = m * (2.0 ** e);
      if (b[31]) f32 = -f32;
    end
  endfunction

  function automatic real rabs(input real v);
    rabs = (v < 0.0) ? -v : v;
  endfunction

  always @(posedge clk) begin
    if (rst_n && out_valid) begin
      if (rabs(f32(out_y) - f32(ex[got])) >
          (rabs(f32(ex[got])) * 1.0e-4 + 1.0e-6)) begin
        errors = errors + 1;
        if (errors < 8)
          $display("  row %0d elem %0d: got %f expected %f",
                   t, got, f32(out_y), f32(ex[got]));
      end
      got = got + 1;
    end
  end

  int fd, code;
  string tag;

  initial begin
    start = 1'b0; acc_valid = 1'b0; acc_last = 1'b0; mul_valid = 1'b0;
    acc_x = '0; mul_x = '0; mul_w = '0;

    fd = $fopen("tb/sv/rmsnorm_vectors.txt", "r");
    if (fd == 0) begin $display("TB ERROR: no vectors"); $fatal(1); end
    code = $fscanf(fd, "%s %h %h", tag, inv_n, eps);
    if (code != 3) begin $display("TB ERROR: bad header"); $fatal(1); end

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    for (t = 0; t < NTEST; t = t + 1) begin
      void'($fscanf(fd, "%s", tag));
      for (i = 0; i < NROW; i = i + 1) void'($fscanf(fd, "%h", xs[i]));
      for (i = 0; i < NROW; i = i + 1) void'($fscanf(fd, "%h", ws[i]));
      for (i = 0; i < NROW; i = i + 1) void'($fscanf(fd, "%h", ex[i]));
      got = 0;

      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;

      for (i = 0; i < NROW; i = i + 1) begin
        acc_valid = 1'b1;
        acc_x     = xs[i];
        acc_last  = (i == NROW-1);
        @(negedge clk);
      end
      acc_valid = 1'b0;
      acc_last  = 1'b0;

      while (!scale_ready) @(negedge clk);

      for (i = 0; i < NROW; i = i + 1) begin
        mul_valid = 1'b1;
        mul_x     = xs[i];
        mul_w     = ws[i];
        @(negedge clk);
      end
      mul_valid = 1'b0;

      repeat (12) @(negedge clk);
      if (got != NROW) begin
        $display("  row %0d: %0d outputs, expected %0d", t, got, NROW);
        errors = errors + 1;
      end
    end
    $fclose(fd);

    if (errors == 0)
      $display("TB_RMSNORM: PASS  (%0d rows of %0d, 0 mismatches)", NTEST, NROW);
    else
      $display("TB_RMSNORM: FAIL  (%0d errors)", errors);
    $finish;
  end

endmodule
