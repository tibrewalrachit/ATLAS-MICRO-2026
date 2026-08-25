#!/usr/bin/env python3
"""Stimulus and expected results for tb_matrix_unit.sv.

Models one output-stationary GEMM pass over the PE array.  Each physical row r
time-multiplexes ACC_N logical M rows, so the array produces an
(ROWS*ACC_N) x COLS output block:

    logical M index = t*ROWS + r        (t is the accumulator/tile id)
    logical N index = c

The issue order is: for each K block, walk the ACC_N tiles.  That is what keeps
a tile from being re-accumulated closer together than the adder latency, and it
is also the natural order for weight reuse -- one B block feeds ACC_N different
A rows before it is retired.

Emits a cycle-by-cycle stimulus file; the RTL's internal skew is not modelled
here, only the edge presentation.
"""
import random
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from atlas_fp import E2M1, E4M3, dot, f32_add   # noqa: E402

N = 32


def pack(codes):
    v = 0
    for i, c in enumerate(codes):
        v |= (c & 0xFF) << (8 * i)
    return v


def main(path, rows=3, cols=4, kblk=6, acc_n=4, seed=4242):
    rng = random.Random(seed)
    fmt_a, fmt_b = E2M1, E4M3         # FP4 weights against FP8 activations
    m_total = rows * acc_n

    hi_a = 16 if fmt_a == E2M1 else 256
    hi_b = 16 if fmt_b == E2M1 else 256

    # A[m][k], B[c][k] and their E8M0 block scales.
    A = [[[rng.randrange(hi_a) for _ in range(N)] for _ in range(kblk)]
         for _ in range(m_total)]
    B = [[[rng.randrange(hi_b) for _ in range(N)] for _ in range(kblk)]
         for _ in range(cols)]
    SA = [[rng.randrange(120, 135) for _ in range(kblk)] for _ in range(m_total)]
    SB = [[rng.randrange(120, 135) for _ in range(kblk)] for _ in range(cols)]

    # Every record is tagged so the testbench can read the file with a single
    # $fscanf loop; untagged comment lines would derail it.
    lines = []
    lines.append("HEADER %d %d %d %d %d %d" % (rows, cols, kblk, acc_n, fmt_a, fmt_b))

    for k in range(kblk):
        for t in range(acc_n):
            ablk = " ".join("%064x" % pack(A[t*rows + r][k]) for r in range(rows))
            asc = " ".join("%02x" % SA[t*rows + r][k] for r in range(rows))
            bblk = " ".join("%064x" % pack(B[c][k]) for c in range(cols))
            bsc = " ".join("%02x" % SB[c][k] for c in range(cols))
            lines.append("CYCLE %d %d %d %s %s %s %s" %
                         (t, 1 if k == 0 else 0, 1 if k == kblk-1 else 0,
                          ablk, asc, bblk, bsc))

    # Expected results, one per (physical row, column, tile).
    for r in range(rows):
        for c in range(cols):
            for t in range(acc_n):
                m = t*rows + r
                acc = 0x00000000
                for k in range(kblk):
                    d = dot(fmt_a, fmt_b, A[m][k], B[c][k], SA[m][k], SB[c][k])
                    acc = f32_add(acc, d)
                lines.append("EXPECT %d %d %d %08x" % (r, c, t, acc))

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %dx%d array, K=%d blocks, %d tiles -> %s"
          % (rows, cols, kblk, acc_n, path))
    print("  %d issue cycles, %d expected results" % (kblk*acc_n, rows*cols*acc_n))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tb/sv/gemm_vectors.txt')
