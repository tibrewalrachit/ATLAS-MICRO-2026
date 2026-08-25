"""Reference model for Kimi Delta Attention's state recurrence.

The recurrence, as published:

    S_t = diag(alpha_t) . S_{t-1}  -  beta_t k_t k_t^T S_{t-1}  +  beta_t k_t v_t^T

with alpha_t and beta_t both produced by a 2-tap convolution over k_t.

Factored for hardware, the last two terms share the k_t outer product:

    u_t   = v_t - S_{t-1}^T k_t                       (a d-vector)
    S_t   = diag(alpha_t) . S_{t-1} + beta_t k_t u_t^T

which turns the update into one matrix-vector product and one rank-1 update
instead of forming k k^T S explicitly -- d^2 work rather than d^3.  Elementwise:

    u[j]     = v[j] - sum_i k[i] * S[i][j]
    S[i][j]  = alpha[i] * S[i][j] + beta * k[i] * u[j]

Note diag(alpha) multiplies from the left, so alpha scales *rows* of S.

All arithmetic is done in binary32 in the same order the RTL uses, so the two
can be compared directly.
"""
import struct


def f32(x):
    """Round a Python float to binary32."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


def conv2(seq, t, w0, w1):
    """2-tap causal convolution over the key sequence: w0*k_t + w1*k_{t-1}."""
    prev = seq[t - 1] if t > 0 else [0.0] * len(seq[t])
    return [f32(f32(w0 * seq[t][i]) + f32(w1 * prev[i])) for i in range(len(seq[t]))]


def sigmoid(x):
    import math
    if x >= 0:
        return f32(1.0 / (1.0 + math.exp(-x)))
    e = math.exp(x)
    return f32(e / (1.0 + e))


def kda_step(S, k, v, alpha, beta):
    """One KDA state update.  S is a list of d rows, each a list of d floats."""
    d = len(k)

    # u = v - S^T k
    u = []
    for j in range(d):
        acc = 0.0
        for i in range(d):
            acc = f32(acc + f32(k[i] * S[i][j]))
        u.append(f32(v[j] - acc))

    # S <- diag(alpha) S + beta k u^T
    Sn = []
    for i in range(d):
        bi = f32(beta * k[i])
        Sn.append([f32(f32(alpha[i] * S[i][j]) + f32(bi * u[j])) for j in range(d)])
    return Sn, u


def kda_read(S, q):
    """Query read-out.  The blog states the recurrence but not the read; this
    is the standard linear-attention form o = S^T q, and is kept separate from
    the update above so the part that is specified can be checked on its own."""
    d = len(q)
    o = []
    for j in range(d):
        acc = 0.0
        for i in range(d):
            acc = f32(acc + f32(q[i] * S[i][j]))
        o.append(acc)
    return o
