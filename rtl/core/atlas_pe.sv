//===========================================================================
// atlas_pe -- one ATLAS processing element
//
// A dot-product engine plus a small bank of binary32 accumulators.  The PE is
// output-stationary: an output tile stays resident in an accumulator while the
// K dimension streams past it one MX block per cycle.
//
// Why there is a bank rather than a single accumulator
// ---------------------------------------------------
// The accumulate step is  acc[id] <- acc[id] + dot_result, and the binary32
// adder takes ACC_LAT cycles.  A single accumulator would therefore only take
// a new block every ACC_LAT cycles and the dot engine would idle.  Holding
// ACC_N tiles and rotating between them keeps the engine fed at one block per
// cycle: the scheduler walks the output tiles round-robin, so by the time an
// accumulator is addressed again its previous update has landed.
//
// The requirement is ACC_N >= ADD_LAT.  At exactly ACC_N == ADD_LAT a tile is
// re-addressed on the same cycle its previous update writes back, and the
// bypass below is what makes that case read the new value rather than the
// stale one.  Closer than ADD_LAT the value simply does not exist yet, so no
// bypass can rescue it; the assertion at the bottom catches a scheduler that
// tries.
//===========================================================================
`include "atlas_defs.svh"

module atlas_pe #(
  parameter int unsigned N     = ATLAS_DOT_N,   // MX block size
  parameter int unsigned ACC_N = 4,             // resident output tiles
  parameter int unsigned ACC_W = $clog2(ACC_N)
) (
  input  logic             clk,
  input  logic             rst_n,

  // Block input.  acc_id selects the output tile this block belongs to;
  // acc_first starts a new tile, acc_last releases it to the output.
  input  logic             in_valid,
  input  logic [1:0]       fmt_a,
  input  logic [1:0]       fmt_b,
  input  logic [N*8-1:0]   a_codes,
  input  logic [N*8-1:0]   b_codes,
  input  logic [7:0]       scale_a,
  input  logic [7:0]       scale_b,
  input  logic [ACC_W-1:0] acc_id,
  input  logic             acc_first,
  input  logic             acc_last,

  output logic             out_valid,
  output logic [ACC_W-1:0] out_id,
  output logic [31:0]      out_f32
);

  localparam int unsigned DOT_LAT = 6;   // atlas_dot_unit pipeline depth
  localparam int unsigned ADD_LAT = 2;   // atlas_fp32_add pipeline depth

  //-------------------------------------------------------------------------
  // Dot-product engine
  //-------------------------------------------------------------------------
  logic        dot_valid;
  logic [31:0] dot_f32;

  atlas_dot_unit #(.N(N)) u_dot (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .fmt_a(fmt_a), .fmt_b(fmt_b),
    .a_codes(a_codes), .b_codes(b_codes),
    .scale_a(scale_a), .scale_b(scale_b),
    .out_valid(dot_valid), .out_f32(dot_f32)
  );

  //-------------------------------------------------------------------------
  // Carry the tile tags alongside the datapath so they arrive with the result.
  //-------------------------------------------------------------------------
  logic [ACC_W-1:0] id_pipe    [0:DOT_LAT-1];
  logic             first_pipe [0:DOT_LAT-1];
  logic             last_pipe  [0:DOT_LAT-1];

  integer d;
  always_ff @(posedge clk) begin
    id_pipe[0]    <= acc_id;
    first_pipe[0] <= acc_first;
    last_pipe[0]  <= acc_last;
    for (d = 1; d < DOT_LAT; d = d + 1) begin
      id_pipe[d]    <= id_pipe[d-1];
      first_pipe[d] <= first_pipe[d-1];
      last_pipe[d]  <= last_pipe[d-1];
    end
  end

  wire [ACC_W-1:0] dot_id    = id_pipe[DOT_LAT-1];
  wire             dot_first = first_pipe[DOT_LAT-1];
  wire             dot_last  = last_pipe[DOT_LAT-1];

  //-------------------------------------------------------------------------
  // Accumulator file with bypass of the in-flight adder results
  //-------------------------------------------------------------------------
  logic [31:0] acc_mem [0:ACC_N-1];

  // Results still inside the adder, newest first.
  logic [ACC_W-1:0] wb_id  [0:ADD_LAT-1];
  logic             wb_vld [0:ADD_LAT-1];
  logic [31:0]      wb_val;              // only the oldest stage has a value

  logic [31:0] acc_rd;
  always_comb begin
    acc_rd = acc_mem[dot_id];
    // The write-back for a tile lands ADD_LAT cycles after its block entered
    // the adder, in the same cycle a round-robin scheduler with
    // ACC_N == ADD_LAT re-addresses it.  acc_mem still holds the old value at
    // that instant, so take the adder output directly.
    if (wb_vld[ADD_LAT-1] && (wb_id[ADD_LAT-1] == dot_id))
      acc_rd = wb_val;
  end

  // A block tagged acc_first starts the tile, so nothing is carried in.
  wire [31:0] addend = dot_first ? 32'h00000000 : acc_rd;

  logic        add_valid;
  logic [31:0] add_result;

  atlas_fp32_add u_acc_add (
    .clk(clk), .rst_n(rst_n),
    .in_valid(dot_valid),
    .a(addend), .b(dot_f32),
    .out_valid(add_valid), .result(add_result)
  );

  assign wb_val = add_result;

  // Tags follow the adder so the write-back knows where to land.
  logic [ACC_W-1:0] add_id  [0:ADD_LAT-1];
  logic             add_lst [0:ADD_LAT-1];

  integer k;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (k = 0; k < ADD_LAT; k = k + 1) wb_vld[k] <= 1'b0;
    end else begin
      wb_vld[0] <= dot_valid;
      for (k = 1; k < ADD_LAT; k = k + 1) wb_vld[k] <= wb_vld[k-1];
    end
  end

  always_ff @(posedge clk) begin
    wb_id[0]  <= dot_id;
    add_id[0] <= dot_id;
    add_lst[0] <= dot_last;
    for (k = 1; k < ADD_LAT; k = k + 1) begin
      wb_id[k]   <= wb_id[k-1];
      add_id[k]  <= add_id[k-1];
      add_lst[k] <= add_lst[k-1];
    end
  end

  always_ff @(posedge clk) begin
    if (add_valid) acc_mem[add_id[ADD_LAT-1]] <= add_result;
  end

  //-------------------------------------------------------------------------
  // Output: a tile leaves the PE when its last block has been accumulated.
  //-------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= add_valid & add_lst[ADD_LAT-1];
  end

  always_ff @(posedge clk) begin
    out_f32 <= add_result;
    out_id  <= add_id[ADD_LAT-1];
  end


  //-------------------------------------------------------------------------
  // Scheduling check (simulation only).  Two accumulate operations on the same
  // tile closer together than ADD_LAT cannot be made correct by any bypass,
  // because the earlier result has not been computed yet.
  //-------------------------------------------------------------------------
`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (dot_valid && !dot_first) begin
      for (int h = 0; h < ADD_LAT-1; h = h + 1) begin
        if (wb_vld[h] && (wb_id[h] == dot_id)) begin
          $display("%0t ERROR atlas_pe: tile %0d re-accumulated %0d cycles after its previous update (needs >= %0d)",
                   $time, dot_id, h+1, ADD_LAT);
          $stop;
        end
      end
    end
  end
`endif

endmodule
