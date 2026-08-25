//===========================================================================
// atlas_vector_unit -- the vector engine of one ATLAS core
//
// A cloud ATLAS core has 480 vector lanes beside its 7680 MACs
// (configs/architecture/chip/cloud/stratum/atlas.yaml).  The engine is split
// in two, because the operations a transformer layer needs are not uniform in
// either cost or rate:
//
//   VEC_N cheap lanes    add, subtract, multiply, min, max, move.  Per element
//                        on every element: residual adds, the MoE gate
//                        multiply, softmax's max pass.  Cheap enough to
//                        replicate.
//   VEC_N/SFU_RATIO SFUs exp, reciprocal, rsqrt, SiLU.  Each carries a
//                        256-entry table, and each appears once per
//                        activation, per row, or per score -- not once per
//                        element of every operand.  Replicating them 480 times
//                        would put more area into transcendentals than into
//                        the MAC array they exist to serve.
//
// The throughput difference is left visible rather than hidden: a full vector
// of transcendentals takes SFU_RATIO passes, and the core's sequencer accounts
// for that.
//===========================================================================
`include "atlas_defs.svh"

module atlas_vector_unit #(
  parameter int unsigned VEC_N     = 32,
  parameter int unsigned SFU_RATIO = 8,
  parameter int unsigned NSFU      = (VEC_N + SFU_RATIO - 1) / SFU_RATIO
) (
  input  logic                clk,
  input  logic                rst_n,

  // ---- elementwise port: VEC_N lanes ----
  input  logic                vec_valid,
  input  logic [2:0]          vec_op,
  input  logic [VEC_N*32-1:0] vec_a,
  input  logic [VEC_N*32-1:0] vec_b,
  output logic                vec_out_valid,
  output logic [VEC_N*32-1:0] vec_y,

  // ---- special-function port: NSFU units ----
  input  logic                sfu_valid,
  input  logic [1:0]          sfu_op,
  input  logic [NSFU*32-1:0]  sfu_x,
  output logic                sfu_out_valid,
  output logic [NSFU*32-1:0]  sfu_y
);

  logic [VEC_N-1:0] lane_vld;
  logic [NSFU-1:0]  sfu_vld;

  genvar i;
  generate
    for (i = 0; i < VEC_N; i++) begin : g_lane
      atlas_vec_lane u_lane (
        .clk(clk), .rst_n(rst_n),
        .in_valid(vec_valid), .op(vec_op),
        .a(vec_a[i*32 +: 32]), .b(vec_b[i*32 +: 32]),
        .out_valid(lane_vld[i]), .y(vec_y[i*32 +: 32])
      );
    end

    for (i = 0; i < NSFU; i++) begin : g_sfu
      atlas_sfu u_sfu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(sfu_valid), .op(sfu_op),
        .x(sfu_x[i*32 +: 32]),
        .out_valid(sfu_vld[i]), .y(sfu_y[i*32 +: 32])
      );
    end
  endgenerate

  // Every lane is driven by the same valid and has the same latency, so lane 0
  // speaks for all of them.
  assign vec_out_valid = lane_vld[0];
  assign sfu_out_valid = sfu_vld[0];

  /* verilator lint_off UNUSEDSIGNAL */
  wire unused_vld = |lane_vld | |sfu_vld;
  /* verilator lint_on UNUSEDSIGNAL */

endmodule
