#!/usr/bin/env python3
"""Stimulus and expected state for tb_kda.sv.

Runs a sequence of tokens through the KDA recurrence, emitting per token the
inputs and the resulting state and read-out.  The state carries across tokens,
so a single wrong update corrupts everything after it -- which is the property
worth testing about a recurrent design.
"""
import random
import struct
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from atlas_kda import f32, kda_step, kda_read, conv2, sigmoid   # noqa: E402


def f2u(x):
    return struct.unpack('<I', struct.pack('<f', f32(x)))[0]


def main(path, d=8, ntok=6, seed=808):
    rng = random.Random(seed)
    lines = ["HEADER %d %d" % (d, ntok)]

    S = [[f32(rng.gauss(0, 0.3)) for _ in range(d)] for _ in range(d)]
    lines.append("STATE0 " + " ".join("%08x" % f2u(S[i][j])
                                      for i in range(d) for j in range(d)))

    ks = [[f32(rng.gauss(0, 0.7)) for _ in range(d)] for _ in range(ntok)]

    for t in range(ntok):
        k = ks[t]
        v = [f32(rng.gauss(0, 0.7)) for _ in range(d)]
        q = [f32(rng.gauss(0, 0.7)) for _ in range(d)]

        # alpha and beta both come from a 2-tap conv over k; alpha is squashed
        # into (0,1) so the state decays rather than diverging, and beta is
        # reduced to a scalar the same way.
        raw_a = conv2(ks, t, 0.7, 0.3)
        alpha = [f32(sigmoid(x)) for x in raw_a]
        raw_b = conv2(ks, t, 0.5, 0.5)
        beta = f32(sigmoid(sum(raw_b) / d))

        S, _ = kda_step(S, k, v, alpha, beta)
        o = kda_read(S, q)

        lines.append("TOKEN "
                     + " ".join("%08x" % f2u(x) for x in k) + " "
                     + " ".join("%08x" % f2u(x) for x in v) + " "
                     + " ".join("%08x" % f2u(x) for x in alpha) + " "
                     + "%08x " % f2u(beta)
                     + " ".join("%08x" % f2u(x) for x in q) + " "
                     + " ".join("%08x" % f2u(S[i][j])
                                for i in range(d) for j in range(d)) + " "
                     + " ".join("%08x" % f2u(x) for x in o))

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %d tokens, d=%d, to %s" % (ntok, d, path))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tb/sv/kda_vectors.txt')
