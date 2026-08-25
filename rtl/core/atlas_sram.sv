//===========================================================================
// atlas_sram -- single-port SRAM macro wrapper
//
// A real build replaces this with a compiled macro from the foundry memory
// compiler.  It is kept behind a wrapper so that swap touches one file, and so
// synthesis can black-box the memory instead of inferring tens of megabits of
// flip-flops: ATLAS gives each core a 4 MB scratchpad, which is a hard macro in
// any real implementation and must never reach the standard-cell mapper.
//
// Define ATLAS_SRAM_BEHAVIOURAL for simulation; leave it undefined for
// synthesis so the instance stays an unresolved black box that the netlist
// hands to the macro during place and route.
//===========================================================================

module atlas_sram #(
  parameter int unsigned WORDS = 1024,
  parameter int unsigned DW    = 512,
  parameter int unsigned AW    = $clog2(WORDS)
) (
  input  logic          clk,
  input  logic          en,
  input  logic          we,
  input  logic [AW-1:0] addr,
  input  logic [DW-1:0] wdata,
  input  logic [DW-1:0] wmask,   // bit-granular write enable
  output logic [DW-1:0] rdata
);

`ifdef ATLAS_SRAM_BEHAVIOURAL
  logic [DW-1:0] mem [0:WORDS-1];

  always_ff @(posedge clk) begin
    if (en) begin
      if (we) mem[addr] <= (wdata & wmask) | (mem[addr] & ~wmask);
      // Read data appears the cycle after the request, as a compiled macro
      // does.  A write returns the old contents on the same address.
      rdata <= mem[addr];
    end
  end
`else
  // Black box: the tool must not infer registers here.  The macro's real
  // behaviour arrives with its Liberty model at link time; nothing in this
  // branch may look like storage to synthesis.
  /* verilator lint_off UNDRIVEN */
  logic [DW-1:0] rdata_bb;
  /* verilator lint_on UNDRIVEN */
  assign rdata = rdata_bb;

  // The ports are genuinely unconnected in black-box mode; sink them so that
  // stays an explicit statement rather than a lint warning per port.
  wire _unused_ok = &{1'b0, clk, en, we, addr, wdata, wmask};
`endif

endmodule
