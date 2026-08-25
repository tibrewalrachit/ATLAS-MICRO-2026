#!/usr/bin/env python3
"""Generate random stimulus + expected results for tb_dot_unit.sv.

Each line is:  fmt_a fmt_b scale_a scale_b a_codes b_codes expected_f32
with a_codes/b_codes as 32 packed bytes (lane 0 in the low byte).
"""
import random
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from atlas_fp import E2M1, E4M3, E5M2, dot, decode

N = 32


def pack(codes):
    v = 0
    for i, c in enumerate(codes):
        v |= (c & 0xFF) << (8 * i)
    return v


def rand_codes(fmt, rng, sparsity=0.0):
    hi = 16 if fmt == E2M1 else 256
    out = []
    for _ in range(N):
        out.append(0 if rng.random() < sparsity else rng.randrange(hi))
    return out


def main(path, n_tests=4000, seed=12345):
    rng = random.Random(seed)
    formats = [E2M1, E4M3, E5M2]
    lines = []
    for t in range(n_tests):
        if t < 200:
            fa = fb = E2M1                      # MXFP4 x MXFP4, the main mode
        elif t < 400:
            fa = fb = E4M3
        elif t < 500:
            fa = fb = E5M2
        elif t < 600:
            fa, fb = E2M1, E4M3                 # mixed weight/activation precision
        else:
            fa, fb = rng.choice(formats), rng.choice(formats)

        # Exercise all-zero blocks, sparse blocks and dense blocks.
        sp = rng.choice([0.0, 0.0, 0.0, 0.25, 0.75, 1.0])
        a = rand_codes(fa, rng, sp)
        b = rand_codes(fb, rng, sp)

        # Scales: mostly nominal, sometimes extreme, rarely the E8M0 NaN code.
        def rscale():
            r = rng.random()
            if r < 0.60:
                return 127
            if r < 0.95:
                return rng.randrange(100, 155)
            if r < 0.99:
                return rng.choice([0, 1, 250, 254])
            return 0xFF
        sa, sb = rscale(), rscale()

        exp = dot(fa, fb, a, b, sa, sb)
        lines.append("%d %d %02x %02x %064x %064x %08x" %
                     (fa, fb, sa, sb, pack(a), pack(b), exp))

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %d vectors to %s" % (len(lines), path))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tb/sv/dot_vectors.txt')
