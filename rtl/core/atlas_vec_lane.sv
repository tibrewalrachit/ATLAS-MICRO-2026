//===========================================================================
// atlas_vec_lane -- one binary32 elementwise lane
//
// The cheap half of the vector unit: add, subtract, multiply, min, max, move.
// These are the operations a layer needs in bulk -- residual adds, the MoE
// gate multiply, attention's value scaling, the max pass of softmax -- and
// they are cheap enough to replicate across every lane.
//
// Transcendentals are deliberately not here; see atlas_sfu.
//
// All operations present a uniform two-cycle latency so a lane's output stream
// stays in step regardless of what it was asked to do.
//===========================================================================
`include "atlas_defs.svh"

module atlas_vec_lane (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [2:0]  op,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        out_valid,
  output logic [31:0] y
);

  // Operation encoding, shared with the core's decoder.
  localparam logic [2:0] VOP_ADD = 3'd0;
  localparam logic [2:0] VOP_SUB = 3'd1;
  localparam logic [2:0] VOP_MUL = 3'd2;
  localparam logic [2:0] VOP_MAX = 3'd3;
  localparam logic [2:0] VOP_MIN = 3'd4;
  localparam logic [2:0] VOP_MOV = 3'd5;

  wire use_add = (op == VOP_ADD) || (op == VOP_SUB);
  wire use_mul = (op == VOP_MUL);

  // Subtract is an add against the sign flip; no separate subtractor.
  wire [31:0] b_eff = (op == VOP_SUB) ? {~b[31], b[30:0]} : b;

  logic        add_v, mul_v;
  logic [31:0] add_y, mul_y;

  atlas_fp32_add u_add (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid & use_add),
    .a(a), .b(b_eff), .out_valid(add_v), .result(add_y)
  );

  atlas_fp32_mul u_mul (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid & use_mul),
    .a(a), .b(b), .out_valid(mul_v), .result(mul_y)
  );

  //-------------------------------------------------------------------------
  // Compare and move.  Ordering uses the monotonic key transform, so the
  // comparison is an unsigned integer one rather than a float compare.
  //-------------------------------------------------------------------------
  function automatic logic [31:0] fkey(input logic [31:0] v);
    fkey = v[31] ? ~v : (v | 32'h8000_0000);
  endfunction

  wire a_ge_b = (fkey(a) >= fkey(b));

  logic [31:0] cmp_y;
  always_comb begin
    unique case (op)
      VOP_MAX: cmp_y = a_ge_b ? a : b;
      VOP_MIN: cmp_y = a_ge_b ? b : a;
      default: cmp_y = a;                       // VOP_MOV
    endcase
  end

  // Two register stages so the compare path matches the arithmetic latency.
  logic [31:0] cmp_q1, cmp_q2;
  logic [2:0]  op_q1, op_q2;
  logic        v_q1, v_q2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v_q1 <= 1'b0;
      v_q2 <= 1'b0;
    end else begin
      v_q1 <= in_valid;
      v_q2 <= v_q1;
    end
  end

  always_ff @(posedge clk) begin
    cmp_q1 <= cmp_y;  cmp_q2 <= cmp_q1;
    op_q1  <= op;     op_q2  <= op_q1;
  end

  always_comb begin
    unique case (op_q2)
      VOP_ADD, VOP_SUB: y = add_y;
      VOP_MUL:          y = mul_y;
      default:          y = cmp_q2;
    endcase
  end

  // add_v and mul_v track the same two-stage latency as v_q2; v_q2 is used so
  // the lane reports a result for compare and move operations too.
  /* verilator lint_off UNUSEDSIGNAL */
  wire unused_v = add_v | mul_v;
  /* verilator lint_on UNUSEDSIGNAL */

  assign out_valid = v_q2;

endmodule
