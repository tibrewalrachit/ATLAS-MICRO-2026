//===========================================================================
// atlas_defs.svh -- Global definitions for the ATLAS RTL implementation
//
// ATLAS is a 3D hybrid-bonding-DRAM (HBDRAM) LLM accelerator.  This header
// carries the numeric-format encodings and the architectural parameters that
// the RTL shares with the ATLAS cycle-level model in simulator/ and with the
// chip descriptions under configs/architecture/.
//
// Style note: yosys (0.33) cannot parse SystemVerilog `package`/`import`, and
// cannot width-infer functions that return a typedef'd struct.  Everything
// here is therefore a `localparam` or a function returning a packed vector so
// that the same sources feed Verilator, Icarus and yosys unmodified.
//===========================================================================
`ifndef ATLAS_DEFS_SVH
`define ATLAS_DEFS_SVH

//---------------------------------------------------------------------------
// Numeric format selector (mirrors atlas_fp_pkg in the Python golden model)
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_FMT_W    = 2;
localparam logic [1:0]  ATLAS_FMT_E2M1 = 2'd0; // FP4, OCP MX  (bias 1)
localparam logic [1:0]  ATLAS_FMT_E4M3 = 2'd1; // FP8, OCP     (bias 7,  no Inf)
localparam logic [1:0]  ATLAS_FMT_E5M2 = 2'd2; // FP8, IEEE-ish (bias 15, has Inf)
localparam logic [1:0]  ATLAS_FMT_BF16 = 2'd3; // BF16 pass-through for norms

//---------------------------------------------------------------------------
// Internal decoded form (IFP).
//
// Every supported input format is decoded to the *same* canonical tuple
//
//     value = (-1)^sign * mant4 * 2^exp
//
// where mant4 is a 4-bit unsigned integer (the significand scaled by 8) and
// exp is a signed exponent.  Making the significand an integer is what lets
// the dot-product datapath below be a pure integer align-and-add tree: for
// MXFP4 the whole reduction is bit-exact, and for FP8 the only loss is the
// bounded alignment window (ATLAS_GUARD_W below).
//
// Packed layout:  { sign, exp[EXP_W-1:0], mant4[3:0], is_zero, is_nan }
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_IFP_EXP_W  = 6;   // signed
localparam int unsigned ATLAS_IFP_MANT_W = 4;   // unsigned, significand*8
localparam int unsigned ATLAS_IFP_W      = 1 + ATLAS_IFP_EXP_W + ATLAS_IFP_MANT_W + 2;

// Field accessors for a packed IFP word
`define ATLAS_IFP_SIGN(w)  (w[ATLAS_IFP_W-1])
`define ATLAS_IFP_EXP(w)   ($signed(w[ATLAS_IFP_W-2 -: ATLAS_IFP_EXP_W]))
`define ATLAS_IFP_MANT(w)  (w[5:2])
`define ATLAS_IFP_ZERO(w)  (w[1])
`define ATLAS_IFP_NAN(w)   (w[0])

//---------------------------------------------------------------------------
// Dot-product datapath widths
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_PROD_MANT_W = 2*ATLAS_IFP_MANT_W;          // 8
localparam int unsigned ATLAS_PROD_EXP_W  = ATLAS_IFP_EXP_W + 1;         // 7, signed
localparam int unsigned ATLAS_GUARD_W     = 20;                          // alignment window
localparam int unsigned ATLAS_ALIGN_W     = ATLAS_PROD_MANT_W + ATLAS_GUARD_W; // 28
// Sum of DOT_N aligned terms, two's complement.  +1 sign, +log2(DOT_N) growth.
localparam int unsigned ATLAS_DOT_N       = 32;   // MX block size (OCP MX = 32)
localparam int unsigned ATLAS_DOT_LOG2N   = 5;
localparam int unsigned ATLAS_SUM_W       = ATLAS_ALIGN_W + ATLAS_DOT_LOG2N + 1;  // 34

//---------------------------------------------------------------------------
// MX shared block scale: E8M0 power-of-two exponent, bias 127, 0xFF = NaN
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_SCALE_W    = 8;
localparam logic [7:0]  ATLAS_SCALE_BIAS = 8'd127;
localparam logic [7:0]  ATLAS_SCALE_NAN  = 8'hFF;

//---------------------------------------------------------------------------
// FP32 accumulator format (IEEE-754 binary32, round-to-nearest-even)
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_F32_W      = 32;
localparam int unsigned ATLAS_F32_MANT_W = 23;
localparam int unsigned ATLAS_F32_EXP_W  = 8;
localparam logic [7:0]  ATLAS_F32_BIAS   = 8'd127;

//---------------------------------------------------------------------------
// ATLAS chip parameters.
//
// Defaults track configs/architecture/chip/cloud/stratum/atlas.yaml (the
// cloud ATLAS design point) and configs/architecture/dram/cloud_1TBps.yaml.
// Every module takes these as `parameter`s so a scaled-down instance can be
// elaborated for simulation and synthesis without editing the sources.
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_FREQ_MHZ   = 1000;
localparam int unsigned ATLAS_CORE_NUM   = 16;
localparam int unsigned ATLAS_MAC_NUM    = 7680;  // MACs per core
localparam int unsigned ATLAS_VEC_NUM    = 480;   // vector lanes per core
localparam int unsigned ATLAS_BUF_KB     = 4096;  // scratchpad per core
localparam int unsigned ATLAS_BUF_RD_BPC = 32768; // bytes/cycle
localparam int unsigned ATLAS_BUF_WR_BPC = 32768;

// The 7680 MACs of a cloud ATLAS core are organised as PE_ROWS x PE_COLS
// dot-product engines of ATLAS_DOT_N lanes: 7680 = 15 * 16 * 32.
localparam int unsigned ATLAS_PE_ROWS    = 15;
localparam int unsigned ATLAS_PE_COLS    = 16;

//---------------------------------------------------------------------------
// HBDRAM: hybrid-bonding DRAM, from the ATLAS ramulator2 patch
//   org preset  HBDRAM_4Gb_1024pin_512col : 1024 DQ, 8192 rows, 512 columns
//   timing      HBDRAM_500Mbps            : tCK 2000 ps, nBL 1, nCL 1
//
// 1024 bits * 500 MT/s = 64 GB/s per channel; 16 channels = 1 TB/s, which is
// the cloud_1TBps design point.  Hybrid bonding is why nCL/nBL are 1: the DRAM
// die sits directly on the logic die, so there is no off-package SerDes.
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_HBD_CHANNELS = 16;
localparam int unsigned ATLAS_HBD_DQ_W     = 1024;         // bits per channel
localparam int unsigned ATLAS_HBD_TX_B     = ATLAS_HBD_DQ_W/8; // 128 B per beat
localparam int unsigned ATLAS_HBD_ROWS     = 8192;
localparam int unsigned ATLAS_HBD_COLS     = 512;
localparam int unsigned ATLAS_HBD_BANKS    = 1;
localparam int unsigned ATLAS_HBD_ROW_W    = 13;
localparam int unsigned ATLAS_HBD_COL_W    = 9;

// Timing preset HBDRAM_500Mbps, in DRAM clocks (tCK = 2 ns).
localparam int unsigned ATLAS_HBD_nBL    = 1;
localparam int unsigned ATLAS_HBD_nCL    = 1;
localparam int unsigned ATLAS_HBD_nRCDRD = 8;
localparam int unsigned ATLAS_HBD_nRCDWR = 8;
localparam int unsigned ATLAS_HBD_nRP    = 6;
localparam int unsigned ATLAS_HBD_nRAS   = 17;
localparam int unsigned ATLAS_HBD_nRC    = 23;
localparam int unsigned ATLAS_HBD_nWR    = 8;
localparam int unsigned ATLAS_HBD_nRTP   = 2;
localparam int unsigned ATLAS_HBD_nCWL   = 1;
localparam int unsigned ATLAS_HBD_nWTR   = 3;
localparam int unsigned ATLAS_HBD_nRTW   = 1;
localparam int unsigned ATLAS_HBD_nRFC   = 45;   // 90 ns @ 4 Gb / 2 ns
localparam int unsigned ATLAS_HBD_nREFI  = 1950; // 3900 ns @ 4 Gb, 85 C

// DRAM command encoding used on the internal command bus
localparam int unsigned ATLAS_DRAM_CMD_W = 3;
localparam logic [2:0]  ATLAS_CMD_NOP    = 3'd0;
localparam logic [2:0]  ATLAS_CMD_ACT    = 3'd1;
localparam logic [2:0]  ATLAS_CMD_PRE    = 3'd2;
localparam logic [2:0]  ATLAS_CMD_RD     = 3'd3;
localparam logic [2:0]  ATLAS_CMD_WR     = 3'd4;
localparam logic [2:0]  ATLAS_CMD_REFAB  = 3'd5;

//---------------------------------------------------------------------------
// NoC: 4x4 mesh, 128-byte flits (configs/architecture/noc/4x4_mesh)
//---------------------------------------------------------------------------
localparam int unsigned ATLAS_NOC_X       = 4;
localparam int unsigned ATLAS_NOC_Y       = 4;
localparam int unsigned ATLAS_FLIT_B      = 128;
localparam int unsigned ATLAS_FLIT_W      = ATLAS_FLIT_B*8;
localparam int unsigned ATLAS_NOC_PORTS   = 5;    // N, E, S, W, Local
localparam int unsigned ATLAS_NOC_VC      = 2;
localparam int unsigned ATLAS_NOC_BUF_DEP = 4;

//---------------------------------------------------------------------------
// Qwen3-235B-A22B (configs/models/qwen3_235b_a22b.json)
//---------------------------------------------------------------------------
localparam int unsigned QWEN_HIDDEN      = 4096;
localparam int unsigned QWEN_LAYERS      = 94;
localparam int unsigned QWEN_Q_HEADS     = 64;
localparam int unsigned QWEN_KV_HEADS    = 4;    // grouped-query attention
localparam int unsigned QWEN_HEAD_DIM    = 128;
localparam int unsigned QWEN_EXPERTS     = 128;
localparam int unsigned QWEN_TOPK        = 8;
localparam int unsigned QWEN_MOE_INTER   = 1536;
localparam int unsigned QWEN_DENSE_INTER = 12288;
localparam int unsigned QWEN_VOCAB       = 151936;
localparam int unsigned QWEN_EXPERT_W    = 7;    // log2(128)
localparam int unsigned QWEN_TOPK_W      = 3;

`endif // ATLAS_DEFS_SVH
