//===========================================================================
// tb_fp32_add -- self-checking test of atlas_fp32_add against binary64.
//
// Vectors come from tb/golden/gen_fadd_vectors.py and cover equal-exponent
// pairs, near-total cancellation, operands far enough apart that only the
// sticky bit matters, the Inf/NaN corners, and uniform random bit patterns.
//===========================================================================
`timescale 1ns/1ps

module tb_fp32_add;

  localparam int unsigned MAXV    = 8192;
  localparam int unsigned LATENCY = 2;

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  logic        in_valid;
  logic [31:0] a, b;
  logic        out_valid;
  logic [31:0] result;

  atlas_fp32_add dut (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
    .a(a), .b(b), .out_valid(out_valid), .result(result)
  );

  logic [31:0] v_a[0:MAXV-1], v_b[0:MAXV-1], v_r[0:MAXV-1];
  int nvec = 0, i_out = 0, errors = 0;

  always @(posedge clk) begin
    if (rst_n && out_valid) begin
      if (result !== v_r[i_out]) begin
        errors <= errors + 1;
        if (errors < 8)
          $display("  MISMATCH vec=%0d  a=%08x b=%08x  got=%08x exp=%08x",
                   i_out, v_a[i_out], v_b[i_out], result, v_r[i_out]);
      end
      i_out <= i_out + 1;
    end
  end

  int    fd, code, i;
  string fname;

  initial begin
    in_valid = 1'b0; a = '0; b = '0;
    if (!$value$plusargs("vectors=%s", fname)) fname = "tb/sv/fadd_vectors.txt";
    fd = $fopen(fname, "r");
    if (fd == 0) begin $display("TB ERROR: cannot open %s", fname); $fatal(1); end
    code = 3;
    while ((nvec < MAXV) && (code == 3)) begin
      code = $fscanf(fd, "%h %h %h", v_a[nvec], v_b[nvec], v_r[nvec]);
      if (code == 3) nvec = nvec + 1;
    end
    $fclose(fd);
    $display("tb_fp32_add: loaded %0d vectors", nvec);

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    for (i = 0; i < nvec; i = i + 1) begin
      @(negedge clk);
      in_valid = 1'b1;
      a = v_a[i];
      b = v_b[i];
    end
    @(negedge clk);
    in_valid = 1'b0;
    repeat (LATENCY + 4) @(negedge clk);

    if ((errors == 0) && (i_out == nvec))
      $display("TB_FP32_ADD: PASS  (%0d vectors, 0 mismatches)", nvec);
    else
      $display("TB_FP32_ADD: FAIL  (%0d vectors, %0d results, %0d mismatches)",
               nvec, i_out, errors);
    $finish;
  end

endmodule
