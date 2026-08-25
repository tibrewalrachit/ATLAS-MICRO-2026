//===========================================================================
// tb_kda -- checks the KDA state recurrence across a token sequence
//
// The state carries from token to token, so one wrong update corrupts every
// token after it.  That is the property worth testing about a recurrent
// design, and it is why this runs a sequence rather than single steps: an
// error in the rank-1 update or in the decay would show up as drift, not as a
// single bad answer.
//
// Both the state and the read-out are compared after every token.
//===========================================================================
`timescale 1ns/1ps

module tb_kda;

  parameter int unsigned D     = 8;
  parameter int unsigned PARTS = 8;
  localparam int unsigned IW   = $clog2(D) + 1;
  localparam int unsigned MAXT = 32;

  logic clk = 1'b0, rst_n = 1'b0;
  always #0.5 clk = ~clk;

  logic              start, busy, done, st_we;
  logic [IW-1:0]     st_row;
  logic [D*32-1:0]   k_vec, v_vec, alpha_vec, q_vec, st_wdata, st_rdata, o_vec;
  logic [31:0]       beta;

  atlas_kda #(.D(D), .PARTS(PARTS)) dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .k_vec(k_vec), .v_vec(v_vec), .alpha_vec(alpha_vec),
    .beta(beta), .q_vec(q_vec),
    .st_we(st_we), .st_row(st_row), .st_wdata(st_wdata), .st_rdata(st_rdata),
    .busy(busy), .done(done), .o_vec(o_vec)
  );

  logic [31:0] s0    [0:D*D-1];
  logic [31:0] t_k   [0:MAXT-1][0:D-1];
  logic [31:0] t_v   [0:MAXT-1][0:D-1];
  logic [31:0] t_a   [0:MAXT-1][0:D-1];
  logic [31:0] t_b   [0:MAXT-1];
  logic [31:0] t_q   [0:MAXT-1][0:D-1];
  logic [31:0] t_S   [0:MAXT-1][0:D*D-1];
  logic [31:0] t_o   [0:MAXT-1][0:D-1];

  int f_d, f_n, ntok = 0, errors = 0, t, i, j;
  int fd, code;
  string tag;

  function automatic real f32(input logic [31:0] b);
    real m; int e, kk;
    if (b[30:23] == 8'd0) f32 = 0.0;
    else begin
      m = 1.0;
      for (kk = 0; kk < 23; kk = kk + 1) if (b[kk]) m = m + (2.0 ** (kk - 23));
      e = int'(b[30:23]) - 127;
      f32 = m * (2.0 ** e);
      if (b[31]) f32 = -f32;
    end
  endfunction

  function automatic real rabs(input real v);
    rabs = (v < 0.0) ? -v : v;
  endfunction

  // A tolerance rather than bit equality: the reference sums in the same
  // order, but the RTL folds partial accumulators, so the two are the same
  // computation with a different (equally valid) association.
  function automatic bit close(input logic [31:0] a, input logic [31:0] b);
    close = rabs(f32(a) - f32(b)) <= (rabs(f32(b)) * 1.0e-4 + 1.0e-5);
  endfunction

  initial begin
    start = 1'b0; st_we = 1'b0; st_row = '0; st_wdata = '0;
    k_vec = '0; v_vec = '0; alpha_vec = '0; q_vec = '0; beta = '0;

    fd = $fopen("tb/sv/kda_vectors.txt", "r");
    if (fd == 0) begin $display("TB ERROR: no vectors"); $fatal(1); end
    code = $fscanf(fd, "%s %d %d", tag, f_d, f_n);
    if (code != 3 || tag != "HEADER") begin $display("TB ERROR: header"); $fatal(1); end
    if (f_d != D) begin
      $display("TB ERROR: vectors are d=%0d, DUT is D=%0d", f_d, D); $fatal(1);
    end

    void'($fscanf(fd, "%s", tag));            // STATE0
    for (i = 0; i < D*D; i = i + 1) void'($fscanf(fd, "%h", s0[i]));

    while (ntok < f_n) begin
      void'($fscanf(fd, "%s", tag));          // TOKEN
      for (i = 0; i < D; i = i + 1)   void'($fscanf(fd, "%h", t_k[ntok][i]));
      for (i = 0; i < D; i = i + 1)   void'($fscanf(fd, "%h", t_v[ntok][i]));
      for (i = 0; i < D; i = i + 1)   void'($fscanf(fd, "%h", t_a[ntok][i]));
      void'($fscanf(fd, "%h", t_b[ntok]));
      for (i = 0; i < D; i = i + 1)   void'($fscanf(fd, "%h", t_q[ntok][i]));
      for (i = 0; i < D*D; i = i + 1) void'($fscanf(fd, "%h", t_S[ntok][i]));
      for (i = 0; i < D; i = i + 1)   void'($fscanf(fd, "%h", t_o[ntok][i]));
      ntok = ntok + 1;
    end
    $fclose(fd);
    $display("tb_kda: d=%0d, %0d tokens", D, ntok);

    repeat (4) @(negedge clk);
    rst_n = 1'b1;
    @(negedge clk);

    // Preload the initial state, one row at a time.
    for (i = 0; i < D; i = i + 1) begin
      st_row = IW'(i);
      for (j = 0; j < D; j = j + 1) st_wdata[j*32 +: 32] = s0[i*D + j];
      st_we = 1'b1;
      @(negedge clk);
    end
    st_we = 1'b0;

    for (t = 0; t < ntok; t = t + 1) begin
      for (i = 0; i < D; i = i + 1) begin
        k_vec[i*32 +: 32]     = t_k[t][i];
        v_vec[i*32 +: 32]     = t_v[t][i];
        alpha_vec[i*32 +: 32] = t_a[t][i];
        q_vec[i*32 +: 32]     = t_q[t][i];
      end
      beta = t_b[t];

      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;

      while (!done) @(negedge clk);

      // read-out
      for (i = 0; i < D; i = i + 1)
        if (!close(o_vec[i*32 +: 32], t_o[t][i])) begin
          errors = errors + 1;
          if (errors < 8)
            $display("  token %0d o[%0d]: got %f expected %f",
                     t, i, f32(o_vec[i*32 +: 32]), f32(t_o[t][i]));
        end

      // state
      @(negedge clk);
      for (i = 0; i < D; i = i + 1) begin
        st_row = IW'(i);
        @(negedge clk);
        for (j = 0; j < D; j = j + 1)
          if (!close(st_rdata[j*32 +: 32], t_S[t][i*D + j])) begin
            errors = errors + 1;
            if (errors < 8)
              $display("  token %0d S[%0d][%0d]: got %f expected %f",
                       t, i, j, f32(st_rdata[j*32 +: 32]), f32(t_S[t][i*D + j]));
          end
      end
    end

    if (errors == 0)
      $display("TB_KDA: PASS  (%0d tokens, d=%0d, state and read-out)", ntok, D);
    else
      $display("TB_KDA: FAIL  (%0d errors)", errors);
    $finish;
  end

endmodule
