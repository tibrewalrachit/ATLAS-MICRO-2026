#!/usr/bin/env python3
"""Generate rtl/gen/atlas_recip_lut.sv and report its accuracy.

Table for 1/m over m in [1,2), which is a binary32 significand.  The result
lies in (0.5, 1], so it is stored scaled by 2^24 to land in (2^23, 2^24] and
share the packing path with everything else: the caller subtracts one from the
exponent and takes the low 23 bits as the significand.

Same shape as the exp2 table -- base plus a linear correction -- but 1/m is
much more curved than 2^f near m = 1, so the segment error is larger for the
same segment count and the generator reports it rather than assuming.
"""
import math
import sys

IDX_W = 8
FRAC_W = 16
SCALE = 1 << 24


def build():
    n = 1 << IDX_W
    base, delta = [], []
    for i in range(n):
        m0 = 1.0 + i / n
        m1 = 1.0 + (i + 1) / n
        lo = SCALE / m0
        hi = SCALE / m1
        d = lo - hi                       # 1/m decreases, so store a magnitude
        # Peak error of a chord under a convex curve, halved and added to the
        # base so the approximation straddles the true curve.
        peak = (SCALE * (1.0 / n) ** 2) / (4.0 * m0 ** 3) / 2.0
        base.append(int(round(lo - peak)))
        delta.append(int(round(d)))
    return base, delta


def evaluate(base, delta):
    n = 1 << IDX_W
    worst, at = 0.0, 0.0
    steps = 1 << 14
    for s in range(steps):
        m = 1.0 + s / steps
        idx = int((m - 1.0) * n)
        frac = int(((m - 1.0) * n - idx) * (1 << FRAC_W))
        approx = (base[idx] - ((delta[idx] * frac) >> FRAC_W)) / SCALE
        exact = 1.0 / m
        rel = abs(approx - exact) / exact
        if rel > worst:
            worst, at = rel, m
    return worst, at


def emit(path, base, delta):
    n = 1 << IDX_W
    bw = max(x.bit_length() for x in base)
    dw = max(x.bit_length() for x in delta)
    with open(path, 'w') as f:
        f.write("""//===========================================================================
// atlas_recip_lut -- GENERATED FILE, do not edit
//
// Regenerate with:  python3 tb/golden/gen_recip_lut.py rtl/gen/atlas_recip_lut.sv
//
// Piecewise-linear table for 1/m, m in [1,2):
//
//     1/m  ~  (base - ((delta * frac) >> %d)) * 2^-24
//
// The subtraction is deliberate: 1/m decreases across each segment, so delta
// is stored as a magnitude and the interpolation walks downwards.
//
// %d segments; the generator reports the worst-case relative error.
//===========================================================================

module atlas_recip_lut (
  input  logic [%d:0]  idx,
  output logic [%d:0]  base,
  output logic [%d:0]  delta
);

  always_comb begin
    case (idx)
""" % (FRAC_W, n, IDX_W - 1, bw - 1, dw - 1))
        for i in range(n):
            f.write("      %d'd%-4d: begin base = %d'd%-9d; delta = %d'd%-7d; end\n"
                    % (IDX_W, i, bw, base[i], dw, delta[i]))
        f.write("""      default:   begin base = %d'd%d; delta = %d'd%d; end
    endcase
  end

endmodule
""" % (bw, base[0], dw, delta[0]))
    return bw, dw


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'rtl/gen/atlas_recip_lut.sv'
    b, d = build()
    rel, at = evaluate(b, d)
    bw, dw = emit(out, b, d)
    print("wrote %s" % out)
    print("  segments      : %d" % (1 << IDX_W))
    print("  base width    : %d bits, delta width: %d bits" % (bw, dw))
    print("  worst rel err : %.3e  (2^-%.1f) at m=%.5f" % (rel, -math.log2(rel), at))
