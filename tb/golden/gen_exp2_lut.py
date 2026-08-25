#!/usr/bin/env python3
"""Generate rtl/gen/atlas_exp2_lut.sv and report its worst-case accuracy.

The table evaluates 2^f for f in [0,1) by linear interpolation:

    2^f  ~  (base[i] + (delta[i] * frac_low >> FRAC_W)) * 2^-23

where i is the top IDX_W bits of f and frac_low is what remains.  base is
2^(i/2^IDX_W) scaled by 2^23, so it lands in [2^23, 2^24) and its low 23 bits
are directly the binary32 significand of the result.

Interpolating a convex function always undershoots, so the table stores base
shifted up by half the maximum segment error.  That converts a one-sided error
of E into a two-sided one of E/2 for free.
"""
import sys


IDX_W = 8            # 256 segments
FRAC_W = 16          # interpolation fraction bits
SCALE = 1 << 23


def build():
    n = 1 << IDX_W
    base, delta = [], []
    for i in range(n):
        lo = 2.0 ** (i / n) * SCALE
        hi = 2.0 ** ((i + 1) / n) * SCALE
        d = hi - lo
        # Half the peak interpolation error of this segment, added to the base
        # so the approximation straddles the true curve instead of sitting
        # under it everywhere.
        peak = (d * (0.6931471805599453 / n)) / 8.0
        base.append(int(round(lo + peak)))
        delta.append(int(round(d)))
    return base, delta


def evaluate(base, delta):
    """Worst absolute and relative error of the table over a dense sweep."""
    n = 1 << IDX_W
    worst_rel = 0.0
    worst_at = 0.0
    steps = 1 << 14
    for s in range(steps):
        f = s / steps
        idx = int(f * n)
        frac_low = int((f * n - idx) * (1 << FRAC_W))
        approx = (base[idx] + ((delta[idx] * frac_low) >> FRAC_W)) / SCALE
        exact = 2.0 ** f
        rel = abs(approx - exact) / exact
        if rel > worst_rel:
            worst_rel, worst_at = rel, f
    return worst_rel, worst_at


def emit(path, base, delta):
    n = 1 << IDX_W
    bw = max(x.bit_length() for x in base)
    dw = max(x.bit_length() for x in delta)
    with open(path, 'w') as f:
        f.write("""//===========================================================================
// atlas_exp2_lut -- GENERATED FILE, do not edit
//
// Regenerate with:  python3 tb/golden/gen_exp2_lut.py rtl/gen/atlas_exp2_lut.sv
//
// Piecewise-linear table for 2^f, f in [0,1):
//
//     2^f  ~  (base + ((delta * frac) >> %d)) * 2^-23
//
// base lands in [2^23, 2^24), so its low 23 bits are directly the binary32
// significand of the result.  Bases are pre-shifted by half a segment's peak
// interpolation error, which centres the approximation on the true curve
// rather than leaving it uniformly low.
//
// %d segments; worst-case relative error is reported by the generator.
//===========================================================================

module atlas_exp2_lut (
  input  logic [%d:0]  idx,
  output logic [%d:0]  base,
  output logic [%d:0]  delta
);

  always_comb begin
    case (idx)
""" % (FRAC_W, n, IDX_W - 1, bw - 1, dw - 1))
        for i in range(n):
            f.write("      %d'd%-4d: begin base = %d'd%-8d; delta = %d'd%-6d; end\n"
                    % (IDX_W, i, bw, base[i], dw, delta[i]))
        f.write("""      default:   begin base = %d'd%d; delta = %d'd%d; end
    endcase
  end

endmodule
""" % (bw, base[0], dw, delta[0]))
    return bw, dw


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'rtl/gen/atlas_exp2_lut.sv'
    b, d = build()
    rel, at = evaluate(b, d)
    bw, dw = emit(out, b, d)
    print("wrote %s" % out)
    print("  segments      : %d" % (1 << IDX_W))
    print("  base width    : %d bits, delta width: %d bits" % (bw, dw))
    print("  worst rel err : %.3e  (2^-%.1f) at f=%.5f"
          % (rel, -__import__('math').log2(rel), at))
