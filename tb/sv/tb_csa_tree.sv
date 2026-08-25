//===========================================================================
// tb_csa_tree -- checks that the carry-save reduction preserves the sum.
//
// The invariant under test is  s_out + c_out == sum(din)  (mod 2^W).  Inputs
// are registered on the negative edge and results sampled on the positive
// edge, so the combinational tree is always given a full half-cycle to settle.
//===========================================================================
`timescale 1ns/1ps

module tb_csa_tree;

  localparam int unsigned N     = 32;
  localparam int unsigned W     = 34;
  localparam int unsigned TERM_W = 29;   // matches atlas_dot_unit's aligned term

  logic clk = 1'b0;
  always #0.5 clk = ~clk;

  logic [N*W-1:0]      din;
  logic signed [W-1:0] s_out, c_out;

  atlas_csa_tree #(.N(N), .W(W)) dut (
    .din(din), .s_out(s_out), .c_out(c_out)
  );

  logic signed [W-1:0] expect_q;
  logic signed [W-1:0] got;
  logic [31:0]         rnd;
  int                  t, i, errors;

  initial begin
    errors = 0;
    din    = '0;
    expect_q = '0;

    for (t = 0; t < 20000; t = t + 1) begin
      @(negedge clk);
      expect_q = '0;
      for (i = 0; i < N; i = i + 1) begin
        // TERM_W-bit signed terms sign-extended to W, exactly as
        // atlas_dot_unit feeds the tree.
        rnd = $random;
        din[i*W +: W] = {{(W-TERM_W){rnd[TERM_W-1]}}, rnd[TERM_W-1:0]};
        expect_q = expect_q + $signed({{(W-TERM_W){rnd[TERM_W-1]}}, rnd[TERM_W-1:0]});
      end

      @(posedge clk);
      got = s_out + c_out;
      if (got !== expect_q) begin
        errors = errors + 1;
        if (errors <= 4)
          $display("  MISMATCH t=%0d got=%0d expected=%0d", t, got, expect_q);
      end
    end

    if (errors == 0)
      $display("TB_CSA_TREE: PASS  (%0d vectors, s+c == sum)", t);
    else
      $display("TB_CSA_TREE: FAIL  (%0d mismatches)", errors);
    $finish;
  end

endmodule
