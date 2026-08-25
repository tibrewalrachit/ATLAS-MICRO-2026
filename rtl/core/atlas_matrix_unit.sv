//===========================================================================
// atlas_matrix_unit -- the MAC array of one ATLAS core
//
// ROWS x COLS processing elements, each holding a 32-lane MX dot-product
// engine.  The cloud ATLAS core is 7680 MACs (see the chip YAML under
// configs/architecture/chip/cloud/stratum/), which is exactly 15 x 16 x 32.
//
// Dataflow
// --------
// Output-stationary and systolic.  Operands are *not* broadcast: a fanout of
// 240 across a millimetre of die is not a wire, it is a repeater tree with its
// own pipeline depth, and pretending otherwise is how an array that simulates
// at 1 GHz fails to close timing.  Instead each operand takes one register hop
// per PE it passes, so every net in the array is a short point-to-point link:
//
//     A operands walk left to right along a row
//     B operands walk top to bottom down a column
//
// That makes PE(r,c) see its A operand c cycles late and its B operand r
// cycles late.  To land matching K blocks in the same PE on the same cycle,
// the edges are pre-skewed -- row r delayed by r, column c delayed by c -- so
// both operands arrive having been delayed by exactly r+c.  The caller is
// therefore free to present block k of every row and column on cycle k, and
// the skew is entirely internal.
//
// Control (tile id, first, last) travels with the A operand so it stays
// aligned with the data it describes.
//===========================================================================
`include "atlas_defs.svh"

module atlas_matrix_unit #(
  parameter int unsigned ROWS  = ATLAS_PE_ROWS,   // 15
  parameter int unsigned COLS  = ATLAS_PE_COLS,   // 16
  parameter int unsigned N     = ATLAS_DOT_N,     // 32 -> 7680 MACs
  parameter int unsigned ACC_N = 4,
  parameter int unsigned ACC_W = $clog2(ACC_N)
) (
  input  logic                  clk,
  input  logic                  rst_n,

  input  logic                  in_valid,
  input  logic [1:0]            fmt_a,
  input  logic [1:0]            fmt_b,
  input  logic [ACC_W-1:0]      acc_id,
  input  logic                  acc_first,
  input  logic                  acc_last,

  // One MX block per row for A, per column for B, presented on the same cycle.
  input  logic [ROWS*N*8-1:0]   a_blocks,
  input  logic [ROWS*8-1:0]     a_scales,
  input  logic [COLS*N*8-1:0]   b_blocks,
  input  logic [COLS*8-1:0]     b_scales,

  // Results, one per PE.
  output logic [ROWS*COLS-1:0]      out_valid,
  output logic [ROWS*COLS*32-1:0]   out_f32,
  output logic [ROWS*COLS*ACC_W-1:0] out_id
);

  localparam int unsigned BLK_W  = N*8;
  // What travels with A: block, scale, valid, first, last, tile id, formats.
  localparam int unsigned ACTL_W = BLK_W + 8 + 3 + ACC_W + 4;
  localparam int unsigned BCTL_W = BLK_W + 8;

  genvar r, c, d;

  //=========================================================================
  // Edge skew: row r by r cycles, column c by c cycles.
  //=========================================================================
  logic [ROWS*ACTL_W-1:0] a_edge;
  logic [COLS*BCTL_W-1:0] b_edge;

  generate
    for (r = 0; r < ROWS; r++) begin : g_askew
      wire [ACTL_W-1:0] raw = {fmt_b, fmt_a, acc_id, acc_last, acc_first, in_valid,
                               a_scales[r*8 +: 8], a_blocks[r*BLK_W +: BLK_W]};
      if (r == 0) begin : g_none
        assign a_edge[0 +: ACTL_W] = raw;
      end else begin : g_delay
        logic [r*ACTL_W-1:0] sr;
        always_ff @(posedge clk) begin
          sr[0 +: ACTL_W] <= raw;
        end
        for (d = 1; d < r; d++) begin : g_stage
          always_ff @(posedge clk) begin
            sr[d*ACTL_W +: ACTL_W] <= sr[(d-1)*ACTL_W +: ACTL_W];
          end
        end
        assign a_edge[r*ACTL_W +: ACTL_W] = sr[(r-1)*ACTL_W +: ACTL_W];
      end
    end

    for (c = 0; c < COLS; c++) begin : g_bskew
      wire [BCTL_W-1:0] raw = {b_scales[c*8 +: 8], b_blocks[c*BLK_W +: BLK_W]};
      if (c == 0) begin : g_none
        assign b_edge[0 +: BCTL_W] = raw;
      end else begin : g_delay
        logic [c*BCTL_W-1:0] sr;
        always_ff @(posedge clk) begin
          sr[0 +: BCTL_W] <= raw;
        end
        for (d = 1; d < c; d++) begin : g_stage
          always_ff @(posedge clk) begin
            sr[d*BCTL_W +: BCTL_W] <= sr[(d-1)*BCTL_W +: BCTL_W];
          end
        end
        assign b_edge[c*BCTL_W +: BCTL_W] = sr[(c-1)*BCTL_W +: BCTL_W];
      end
    end
  endgenerate

  //=========================================================================
  // The array.  a_bus[r][c] and b_bus[r][c] are the operands entering PE(r,c);
  // each is one register hop from its neighbour.
  //=========================================================================
  logic [ROWS*COLS*ACTL_W-1:0] a_bus;
  logic [ROWS*COLS*BCTL_W-1:0] b_bus;

  generate
    for (r = 0; r < ROWS; r++) begin : g_row
      for (c = 0; c < COLS; c++) begin : g_col
        localparam int unsigned P = r*COLS + c;

        // --- operand arrival ---
        if (c == 0) begin : g_a_edge
          assign a_bus[P*ACTL_W +: ACTL_W] = a_edge[r*ACTL_W +: ACTL_W];
        end else begin : g_a_hop
          logic [ACTL_W-1:0] q;
          always_ff @(posedge clk) q <= a_bus[(P-1)*ACTL_W +: ACTL_W];
          assign a_bus[P*ACTL_W +: ACTL_W] = q;
        end

        if (r == 0) begin : g_b_edge
          assign b_bus[P*BCTL_W +: BCTL_W] = b_edge[c*BCTL_W +: BCTL_W];
        end else begin : g_b_hop
          logic [BCTL_W-1:0] q;
          always_ff @(posedge clk) q <= b_bus[(P-COLS)*BCTL_W +: BCTL_W];
          assign b_bus[P*BCTL_W +: BCTL_W] = q;
        end

        // --- unpack ---
        wire [ACTL_W-1:0] av = a_bus[P*ACTL_W +: ACTL_W];
        wire [BCTL_W-1:0] bv = b_bus[P*BCTL_W +: BCTL_W];

        wire [BLK_W-1:0]  pe_a     = av[0            +: BLK_W];
        wire [7:0]        pe_sa    = av[BLK_W        +: 8];
        wire              pe_vld   = av[BLK_W+8];
        wire              pe_first = av[BLK_W+9];
        wire              pe_last  = av[BLK_W+10];
        wire [ACC_W-1:0]  pe_id    = av[BLK_W+11     +: ACC_W];
        wire [1:0]        pe_fa    = av[BLK_W+11+ACC_W +: 2];
        wire [1:0]        pe_fb    = av[BLK_W+13+ACC_W +: 2];

        wire [BLK_W-1:0]  pe_b     = bv[0     +: BLK_W];
        wire [7:0]        pe_sb    = bv[BLK_W +: 8];

        atlas_pe #(.N(N), .ACC_N(ACC_N)) u_pe (
          .clk(clk), .rst_n(rst_n),
          .in_valid(pe_vld),
          .fmt_a(pe_fa), .fmt_b(pe_fb),
          .a_codes(pe_a), .b_codes(pe_b),
          .scale_a(pe_sa), .scale_b(pe_sb),
          .acc_id(pe_id), .acc_first(pe_first), .acc_last(pe_last),
          .out_valid(out_valid[P]),
          .out_id   (out_id[P*ACC_W +: ACC_W]),
          .out_f32  (out_f32[P*32 +: 32])
        );
      end
    end
  endgenerate

endmodule
