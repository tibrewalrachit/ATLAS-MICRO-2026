//===========================================================================
// atlas_fp_decode -- decode one FP4/FP8 code into the canonical IFP tuple
//
//     value = (-1)^sign * mant4 * 2^exp        (mant4 is an integer, 0..15)
//
// Holding the significand as a plain integer is the trick that makes the
// downstream dot product an integer align-and-add tree.  Subnormals need no
// special case beyond clamping the exponent to its minimum normal value,
// because the implicit leading one is simply dropped from mant4.
//
// Combinational.  One instance per multiplier lane.
//===========================================================================
`include "atlas_defs.svh"

module atlas_fp_decode (
  input  logic [1:0]              fmt,   // ATLAS_FMT_*
  input  logic [7:0]              code,  // FP4 uses code[3:0]
  output logic [ATLAS_IFP_W-1:0]  ifp
);

  logic                             s;
  logic signed [ATLAS_IFP_EXP_W-1:0] e;
  logic [ATLAS_IFP_MANT_W-1:0]      m;
  logic                             z, n;

  // E2M1 fields
  logic       e1_s;  logic [1:0] e1_e;  logic       e1_m;
  // E4M3 fields
  logic       e4_s;  logic [3:0] e4_e;  logic [2:0] e4_m;
  // E5M2 fields
  logic       e5_s;  logic [4:0] e5_e;  logic [1:0] e5_m;

  assign {e1_s, e1_e, e1_m} = code[3:0];
  assign {e4_s, e4_e, e4_m} = code[7:0];
  assign {e5_s, e5_e, e5_m} = code[7:0];

  always_comb begin
    // Default: zero
    s = 1'b0; e = '0; m = '0; z = 1'b1; n = 1'b0;

    unique case (fmt)
      //-------------------------------------------------------------------
      // FP4 E2M1, bias 1.  Value set +/-{0, .5, 1, 1.5, 2, 3, 4, 6}.
      // normal   : (1 + M/2) * 2^(E-1)  ->  mant4 = 8 + 4M, exp = E - 4
      // subnormal: (M/2)     * 2^0      ->  mant4 = 4M,     exp = 1 - 4
      //-------------------------------------------------------------------
      ATLAS_FMT_E2M1: begin
        s = e1_s;
        z = (e1_e == 2'd0) && (e1_m == 1'b0);
        n = 1'b0;                                   // E2M1 has no Inf/NaN
        m = (e1_e == 2'd0) ? {1'b0, e1_m, 2'b00}    //      4*M
                           : {1'b1, e1_m, 2'b00};   //  8 + 4*M
        e = $signed({4'd0, (e1_e == 2'd0) ? 2'd1 : e1_e}) - $signed(6'sd4);
      end

      //-------------------------------------------------------------------
      // FP8 E4M3, OCP variant: bias 7, no Inf, NaN == S.1111.111, max 448.
      // normal   : (1 + M/8) * 2^(E-7)  ->  mant4 = 8 + M, exp = E - 10
      // subnormal: (M/8)     * 2^-6     ->  mant4 = M,     exp = 1 - 10
      //-------------------------------------------------------------------
      ATLAS_FMT_E4M3: begin
        s = e4_s;
        z = (e4_e == 4'd0) && (e4_m == 3'd0);
        n = (e4_e == 4'hF) && (e4_m == 3'h7);
        m = (e4_e == 4'd0) ? {1'b0, e4_m} : {1'b1, e4_m};
        e = $signed({2'd0, (e4_e == 4'd0) ? 4'd1 : e4_e}) - $signed(6'sd10);
      end

      //-------------------------------------------------------------------
      // FP8 E5M2: bias 15, IEEE-style Inf/NaN at E==31.
      // normal   : (1 + M/4) * 2^(E-15) ->  mant4 = 8 + 2M, exp = E - 18
      // subnormal: (M/4)     * 2^-14    ->  mant4 = 2M,     exp = 1 - 18
      //
      // Inf and NaN are both reported through the `nan` flag: an inference
      // datapath saturates rather than propagating a distinct infinity, and
      // carrying a separate Inf state costs area in every lane for no gain.
      //-------------------------------------------------------------------
      ATLAS_FMT_E5M2: begin
        s = e5_s;
        z = (e5_e == 5'd0) && (e5_m == 2'd0);
        n = (e5_e == 5'd31);
        m = (e5_e == 5'd0) ? {1'b0, e5_m, 1'b0} : {1'b1, e5_m, 1'b0};
        e = $signed({1'b0, (e5_e == 5'd0) ? 5'd1 : e5_e}) - $signed(6'sd18);
      end

      //-------------------------------------------------------------------
      // BF16 is never fed to the low-precision multiplier array; it reaches
      // the vector unit directly.  Decoding it here would need a wider
      // significand, so the lane reports zero and flags NaN to make an
      // accidental connection visible in simulation instead of silent.
      //-------------------------------------------------------------------
      ATLAS_FMT_BF16: begin
        s = 1'b0; e = '0; m = '0; z = 1'b1; n = 1'b1;
      end
      default: ;
    endcase
  end

  assign ifp = {s, e, m, z, n};

endmodule
