"""Bit-accurate reference model for the ATLAS low-precision datapath.

Mirrors rtl/fp/atlas_fp_decode.sv and rtl/fp/atlas_dot_unit.sv exactly, so the
testbenches can compare RTL output against it bit for bit.  All arithmetic here
is done with Python integers and Fraction, never floats, so the reference is
independent of host FPU behaviour.
"""

from fractions import Fraction

E2M1, E4M3, E5M2, BF16 = 0, 1, 2, 3

# Widths -- keep in sync with rtl/include/atlas_defs.svh
IFP_MANT_W = 4
PROD_MANT_W = 8
GUARD_W = 20
ALIGN_W = PROD_MANT_W + GUARD_W          # 28
DOT_N = 32
SUM_W = ALIGN_W + 5 + 1                  # 34
SCALE_BIAS = 127


def decode(fmt, code):
    """Return (sign, exp, mant4, is_zero, is_nan) with value = (-1)^s * mant4 * 2^exp."""
    if fmt == E2M1:
        s = (code >> 3) & 1
        e = (code >> 1) & 3
        m = code & 1
        z = (e == 0 and m == 0)
        mant = (m << 2) if e == 0 else (8 | (m << 2))
        exp = (1 if e == 0 else e) - 4
        return (s, exp, mant, z, False)
    if fmt == E4M3:
        s = (code >> 7) & 1
        e = (code >> 3) & 0xF
        m = code & 0x7
        z = (e == 0 and m == 0)
        n = (e == 0xF and m == 0x7)
        mant = m if e == 0 else (8 | m)
        exp = (1 if e == 0 else e) - 10
        return (s, exp, mant, z, n)
    if fmt == E5M2:
        s = (code >> 7) & 1
        e = (code >> 2) & 0x1F
        m = code & 0x3
        z = (e == 0 and m == 0)
        n = (e == 31)
        mant = (m << 1) if e == 0 else (8 | (m << 1))
        exp = (1 if e == 0 else e) - 18
        return (s, exp, mant, z, n)
    raise ValueError("BF16 is not a multiplier-array input format")


def to_fraction(fmt, code):
    """Exact rational value of a code, for checking `decode` itself."""
    s, exp, mant, z, n = decode(fmt, code)
    if z:
        return Fraction(0)
    if n:
        return None
    v = Fraction(mant) * Fraction(2) ** exp
    return -v if s else v


def dot(fmt_a, fmt_b, a_codes, b_codes, scale_a=SCALE_BIAS, scale_b=SCALE_BIAS):
    """MX block dot product; returns the FP32 bit pattern the RTL must produce.

    Reproduces the hardware exactly: decode, integer multiply, align every
    product to the block's maximum product exponent inside a GUARD_W window,
    sum in two's complement, then normalise and round to nearest-even.
    """
    n = len(a_codes)
    assert n == len(b_codes) <= DOT_N

    if scale_a == 0xFF or scale_b == 0xFF:
        return 0x7FC00000                                  # NaN block scale

    prods = []          # (sign, exp, mant) for non-zero lanes
    any_nan = False
    for ca, cb in zip(a_codes, b_codes):
        sa, ea, ma, za, na = decode(fmt_a, ca)
        sb, eb, mb, zb, nb = decode(fmt_b, cb)
        if na or nb:
            any_nan = True
        if za or zb:
            continue
        prods.append((sa ^ sb, ea + eb, ma * mb))
    if any_nan:
        return 0x7FC00000
    if not prods:
        return 0x00000000

    max_exp = max(p[1] for p in prods)

    acc = 0
    for sgn, exp, mant in prods:
        sh = max_exp - exp
        term = (mant << GUARD_W) >> sh if sh < ALIGN_W else 0
        acc += -term if sgn else term

    if acc == 0:
        return 0x00000000

    # value = acc * 2^(max_exp - GUARD_W) * 2^(scale_a-127) * 2^(scale_b-127)
    exp_off = max_exp - GUARD_W + (scale_a - SCALE_BIAS) + (scale_b - SCALE_BIAS)
    return _pack_f32(acc, exp_off)


def _pack_f32(acc, exp_off):
    """Normalise integer `acc` scaled by 2^exp_off into IEEE-754 binary32 (RNE)."""
    sign = 1 if acc < 0 else 0
    mag = -acc if acc < 0 else acc

    msb = mag.bit_length() - 1                 # index of leading one
    unbiased = msb + exp_off                   # value = 1.f * 2^unbiased

    # Extract 24 significand bits (implicit 1 + 23) with guard/round/sticky.
    if msb > 23:
        shift = msb - 23
        frac = mag >> shift
        rem = mag & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (frac & 1)):
            frac += 1
            if frac >> 24:                     # rounding overflowed into 2.0
                frac >>= 1
                unbiased += 1
    else:
        frac = mag << (23 - msb)               # exact, no rounding needed

    biased = unbiased + 127
    if biased >= 255:
        return (sign << 31) | 0x7F800000       # overflow -> Inf
    if biased <= 0:
        return sign << 31                      # underflow -> signed zero
    return (sign << 31) | (biased << 23) | (frac & 0x7FFFFF)


def f32_to_float(bits):
    import struct
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]


# ---------------------------------------------------------------------------
# binary32 helpers, used by the GEMM reference to accumulate block results in
# exactly the order and precision the hardware does.
# ---------------------------------------------------------------------------
import math as _math
import struct as _struct


def f32_add(ua, ub):
    """Bit pattern of the binary32 sum of two binary32 bit patterns.

    Adding in Python floats (binary64) and rounding once to binary32 is
    correctly rounded here: binary64 has 53 significand bits and the safe
    double-rounding bound for binary32 addition is 2*24 + 2 = 50.
    """
    fa = _struct.unpack('<f', _struct.pack('<I', ua & 0xFFFFFFFF))[0]
    fb = _struct.unpack('<f', _struct.pack('<I', ub & 0xFFFFFFFF))[0]
    if _math.isnan(fa) or _math.isnan(fb):
        return 0x7FC00000
    if _math.isinf(fa) and _math.isinf(fb) and fa != fb:
        return 0x7FC00000
    r = fa + fb
    if _math.isnan(r):
        return 0x7FC00000
    try:
        u = _struct.unpack('<I', _struct.pack('<f', r))[0]
    except OverflowError:
        return 0xFF800000 if r < 0 else 0x7F800000
    # The hardware flushes binary32 subnormal results to a signed zero.
    if ((u >> 23) & 0xFF) == 0 and (u & 0x7FFFFF) != 0:
        return u & 0x80000000
    return u


def gemm_block_accumulate(fmt_a, fmt_b, a_blocks, b_blocks, scales_a, scales_b):
    """Reference for one output element: accumulate MX block dot products.

    Sums in binary32 in issue order, starting from +0, which is what atlas_pe
    does (the first block is added to zero rather than special-cased).
    """
    acc = 0x00000000
    for k in range(len(a_blocks)):
        d = dot(fmt_a, fmt_b, a_blocks[k], b_blocks[k], scales_a[k], scales_b[k])
        acc = f32_add(acc, d)
    return acc
