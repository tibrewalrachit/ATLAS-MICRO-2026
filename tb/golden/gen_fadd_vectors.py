#!/usr/bin/env python3
"""Stimulus and expected results for tb_fp32_add.sv.

The reference adds in Python floats (binary64) and rounds the result to
binary32.  That double rounding is safe here: binary64 carries 53 significand
bits and the safe-double-rounding bound for a binary32 addition is
2*24 + 2 = 50, so rounding via binary64 always lands on the same binary32 the
correctly rounded sum would.

Cases where either input or the true result is subnormal are skipped: the RTL
flushes those by design (see the module header), so checking them against a
reference that does not would be testing a deliberate difference.
"""
import random
import struct
import sys
import math


def f2u(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def u2f(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


def is_subnormal(u):
    return ((u >> 23) & 0xFF) == 0 and (u & 0x7FFFFF) != 0


def round_to_f32(x):
    """Bit pattern of x rounded to binary32; saturates to Inf on overflow."""
    if math.isnan(x):
        return 0x7FC00000
    try:
        return f2u(x)
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def main(path, n=6000, seed=99):
    rng = random.Random(seed)
    out = []

    def emit(ua, ub):
        if is_subnormal(ua) or is_subnormal(ub):
            return
        fa, fb = u2f(ua), u2f(ub)
        if math.isinf(fa) and math.isinf(fb) and (fa != fb):
            ur = 0x7FC00000                       # Inf - Inf
        elif math.isnan(fa) or math.isnan(fb):
            ur = 0x7FC00000
        else:
            ur = round_to_f32(fa + fb)
        if is_subnormal(ur):
            return
        out.append("%08x %08x %08x" % (ua, ub, ur))

    corners = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000,
               0xFF800000, 0x7FC00000, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,
               0x40000000, 0xC0000000, 0x3F7FFFFF, 0x41200000]
    for x in corners:
        for y in corners:
            emit(x, y)

    while len(out) < n:
        mode = rng.randrange(4)
        if mode == 0:                              # equal exponents
            ea = rng.randrange(1, 255)
            ua = (rng.randrange(2) << 31) | (ea << 23) | rng.getrandbits(23)
            ub = (rng.randrange(2) << 31) | (ea << 23) | rng.getrandbits(23)
        elif mode == 1:                            # near-total cancellation
            ua = (rng.randrange(2) << 31) | (rng.randrange(1, 255) << 23) | rng.getrandbits(23)
            ub = ua ^ 0x80000000
            if rng.randrange(2):
                ub ^= rng.randrange(1, 4)
        elif mode == 2:                            # far apart: exercises sticky
            ea = rng.randrange(1, 255)
            eb = max(1, min(254, ea + rng.randrange(-60, 60)))
            ua = (rng.randrange(2) << 31) | (ea << 23) | rng.getrandbits(23)
            ub = (rng.randrange(2) << 31) | (eb << 23) | rng.getrandbits(23)
        else:
            ua = rng.getrandbits(32)
            ub = rng.getrandbits(32)
        emit(ua, ub)

    out = out[:n]
    with open(path, 'w') as f:
        f.write("\n".join(out) + "\n")
    print("wrote %d vectors to %s" % (len(out), path))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tb/sv/fadd_vectors.txt')
