#!/usr/bin/env python3
"""Reference top-k routing for tb_moe_router.sv.

Deliberately follows the *long* route -- softmax over all 128 logits, top-k of
that, then renormalise -- which is what a framework computes.  The RTL takes
the short route.  Agreeing here is the point.
"""
import math
import random
import struct
import sys


def f2u(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def main(path, ntok=32, nexp=128, topk=8, seed=31337):
    rng = random.Random(seed)
    lines = []
    for _ in range(ntok):
        # Round-trip through binary32 so the reference sees exactly the values
        # the RTL is given.
        logits = [struct.unpack('<f', struct.pack('<f', rng.gauss(0, 4)))[0]
                  for _ in range(nexp)]

        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        tot = sum(ex)
        sm = [e / tot for e in ex]

        order = sorted(range(nexp), key=lambda i: (-sm[i], i))[:topk]
        wsum = sum(sm[i] for i in order)
        w = [sm[i] / wsum for i in order]

        lines.append("TOKEN " + " ".join("%08x" % f2u(v) for v in logits)
                     + " " + " ".join(str(i) for i in order)
                     + " " + " ".join("%08x" % f2u(x) for x in w))

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %d tokens (%d experts, top-%d) to %s" % (ntok, nexp, topk, path))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tb/sv/moe_vectors.txt')
