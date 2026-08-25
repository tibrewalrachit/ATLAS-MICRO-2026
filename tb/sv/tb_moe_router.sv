//===========================================================================
// tb_moe_router -- checks top-8-of-128 selection and weight normalisation
//
// Compared against the reference path a framework would take: softmax over
// all 128 logits, top-k of that, then renormalise.  The RTL takes the short
// route (top-k of the raw logits, exponentials of only those 8), so agreement
// here is what shows the two are the same computation.
//===========================================================================
`timescale 1ns/1ps

module tb_moe_router;

  localparam int unsigned NEXP  = 128;
  localparam int unsigned TOPK  = 8;
  localparam int unsigned IDX_W = $clog2(NEXP);
  localparam int unsigned MAXT  = 64;

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  logic                  start, lg_valid, done;
  logic [31:0]           lg_data;
  logic [TOPK*IDX_W-1:0] sel_idx;
  logic [TOPK*32-1:0]    sel_w;

  atlas_moe_router #(.NEXP(NEXP), .TOPK(TOPK)) dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .lg_valid(lg_valid), .lg_data(lg_data),
    .done(done), .sel_idx(sel_idx), .sel_w(sel_w)
  );

  logic [31:0] logits [0:MAXT-1][0:NEXP-1];
  int          e_idx  [0:MAXT-1][0:TOPK-1];
  logic [31:0] e_w    [0:MAXT-1][0:TOPK-1];

  int ntok = 0, errors = 0, t, i;
  int fd, code;
  string tag, fname;

  // Weights are compared with a tolerance: the RTL's exponential and
  // reciprocal are table-based (about 2^-19 each), so bit equality with a
  // binary64 reference is not the right bar.
  // Decoded explicitly rather than via $bitstoshortreal, which Verilator does
  // not support; this is testbench-only and needs no hardware equivalent.
  function automatic real f32(input logic [31:0] b);
    real m;
    int  e, i;
    if (b[30:23] == 8'd0) begin
      f32 = 0.0;                       // zero or subnormal, both flushed
    end else begin
      m = 1.0;
      for (i = 0; i < 23; i = i + 1)
        if (b[i]) m = m + (2.0 ** (i - 23));
      e   = int'(b[30:23]) - 127;
      f32 = m * (2.0 ** e);
      if (b[31]) f32 = -f32;
    end
  endfunction

  function automatic real rabs(input real v);
    rabs = (v < 0.0) ? -v : v;
  endfunction

  initial begin
    start = 1'b0; lg_valid = 1'b0; lg_data = '0;

    if (!$value$plusargs("vectors=%s", fname)) fname = "tb/sv/moe_vectors.txt";
    fd = $fopen(fname, "r");
    if (fd == 0) begin $display("TB ERROR: cannot open %s", fname); $fatal(1); end

    code = 1;
    while (code == 1) begin
      code = $fscanf(fd, "%s", tag);
      if (code == 1) begin
        if (tag == "TOKEN") begin
          for (i = 0; i < NEXP; i = i + 1)
            void'($fscanf(fd, "%h", logits[ntok][i]));
          for (i = 0; i < TOPK; i = i + 1)
            void'($fscanf(fd, "%d", e_idx[ntok][i]));
          for (i = 0; i < TOPK; i = i + 1)
            void'($fscanf(fd, "%h", e_w[ntok][i]));
          ntok = ntok + 1;
        end
      end
    end
    $fclose(fd);
    $display("tb_moe_router: %0d tokens", ntok);

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    for (t = 0; t < ntok; t = t + 1) begin
      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;
      for (i = 0; i < NEXP; i = i + 1) begin
        lg_valid = 1'b1;
        lg_data  = logits[t][i];
        @(negedge clk);
      end
      lg_valid = 1'b0;

      while (!done) @(negedge clk);

      for (i = 0; i < TOPK; i = i + 1) begin
        if (int'(sel_idx[i*IDX_W +: IDX_W]) !== e_idx[t][i]) begin
          errors = errors + 1;
          if (errors < 8)
            $display("  token %0d slot %0d: expert %0d, expected %0d",
                     t, i, sel_idx[i*IDX_W +: IDX_W], e_idx[t][i]);
        end
        if (rabs(f32(sel_w[i*32 +: 32]) - f32(e_w[t][i])) > 1.0e-5) begin
          errors = errors + 1;
          if (errors < 8)
            $display("  token %0d slot %0d: weight %f, expected %f",
                     t, i, f32(sel_w[i*32 +: 32]), f32(e_w[t][i]));
        end
      end
      @(negedge clk);
    end

    if (errors == 0)
      $display("TB_MOE_ROUTER: PASS  (%0d tokens, %0d experts, top-%0d)",
               ntok, NEXP, TOPK);
    else
      $display("TB_MOE_ROUTER: FAIL  (%0d errors)", errors);
    $finish;
  end

endmodule
