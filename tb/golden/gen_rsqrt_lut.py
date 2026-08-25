#!/usr/bin/env python3
"""Generate rtl/gen/atlas_rsqrt_lut.sv and report its accuracy.

1/sqrt(x) needs the exponent halved, so its parity matters: for x = m*2^e with
m in [1,2),

    e even :  1/sqrt(x) = (1/sqrt(m))  * 2^(-e/2)
    e odd  :  1/sqrt(x) = (1/sqrt(2m)) * 2^(-(e-1)/2)

Both 1/sqrt(m) and 1/sqrt(2m) land in (0.5, 1], so one table covering [1,4)
serves both, indexed by the exponent's low bit followed by the significand's
top bits.  Values are scaled by 2^24 so the result is 1.f * 2^-1 and packs with
a single exponent adjustment.
"""
import math
import sys

MIDX_W = 8                       # significand bits used for the index
IDX_W = MIDX_W + 1               # plus the exponent parity bit
FRAC_W = 16
SCALE = 1 << 24


def build():
    nm = 1 << MIDX_W
    base, delta = [], []
    for parity in (0, 1):
        for i in range(nm):
            lo_m = (1.0 + i / nm) * (2.0 if parity else 1.0)
            hi_m = (1.0 + (i + 1) / nm) * (2.0 if parity else 1.0)
            lo = SCALE / math.sqrt(lo_m)
            hi = SCALE / math.sqrt(hi_m)
            d = lo - hi                       # decreasing, store a magnitude
            step = hi_m - lo_m
            # Peak chord error under a convex curve, halved and folded into
            # the base so the fit straddles the curve.
            peak = (SCALE * 0.75 * step * step / (4.0 * lo_m ** 2.5)) / 2.0
            base.append(int(round(lo - peak)))
            delta.append(int(round(d)))
    return base, delta


def evaluate(base, delta):
    worst, at = 0.0, 0.0
    nm = 1 << MIDX_W
    for parity in (0, 1):
        for s in range(1 << 13):
            m = 1.0 + s / (1 << 13)
            x = m * (2.0 if parity else 1.0)
            i = int((m - 1.0) * nm)
            frac = int(((m - 1.0) * nm - i) * (1 << FRAC_W))
            k = parity * nm + i
            approx = (base[k] - ((delta[k] * frac) >> FRAC_W)) / SCALE
            exact = 1.0 / math.sqrt(x)
            rel = abs(approx - exact) / exact
            if rel > worst:
                worst, at = rel, x
    return worst, at


def emit(path, base, delta):
    bw = max(x.bit_length() for x in base)
    dw = max(x.bit_length() for x in delta)
    with open(path, 'w') as f:
        f.write("""//===========================================================================
// atlas_rsqrt_lut -- GENERATED FILE, do not edit
//
// Regenerate with:  python3 tb/golden/gen_rsqrt_lut.py rtl/gen/atlas_rsqrt_lut.sv
//
// Piecewise-linear table for 1/sqrt(v) over v in [1,4):
//
//     1/sqrt(v)  ~  (base - ((delta * frac) >> %d)) * 2^-24
//
// idx is {exponent parity, significand[22:15]}.  Covering [1,4) rather than
// [1,2) is what lets a single table serve both exponent parities: an odd
// exponent is absorbed by doubling the significand.
//
// %d entries; the generator reports the worst-case relative error.
//===========================================================================

module atlas_rsqrt_lut (
  input  logic [%d:0]  idx,
  output logic [%d:0]  base,
  output logic [%d:0]  delta
);

  always_comb begin
    case (idx)
""" % (FRAC_W, len(base), IDX_W - 1, bw - 1, dw - 1))
        for i in range(len(base)):
            f.write("      %d'd%-4d: begin base = %d'd%-9d; delta = %d'd%-7d; end\n"
                    % (IDX_W, i, bw, base[i], dw, delta[i]))
        f.write("""      default:   begin base = %d'd%d; delta = %d'd%d; end
    endcase
  end

endmodule
""" % (bw, base[0], dw, delta[0]))
    return bw, dw


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'rtl/gen/atlas_rsqrt_lut.sv'
    b, d = build()
    rel, at = evaluate(b, d)
    bw, dw = emit(out, b, d)
    print("wrote %s" % out)
    print("  entries       : %d (2 parities x %d)" % (len(b), 1 << MIDX_W))
    print("  base width    : %d bits, delta width: %d bits" % (bw, dw))
    print("  worst rel err : %.3e  (2^-%.1f) at v=%.5f" % (rel, -math.log2(rel), at))
