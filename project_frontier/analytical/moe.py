"""Small-batch MoE expert-reuse models (Part VII).

Three models of routed-expert weight traffic per decode step for batch B:
  1. none:        every (user, expert) invocation streams weights.
  2. statistical: expected unique experts E[U] = sum_e 1-(1-p_e)^B, evaluated
                  for uniform routing (p_e = k/E) or a skewed distribution.
  3. trace:       empirical expert-ID traces (per token, per layer) drive
                  overlap; synthetic Zipf traces provided (real routing traces
                  for V4-Flash/M3 are not public — labeled as synthetic).
"""
import math, random
from typing import List, Optional


def unique_experts_none(n_experts: int, top_k: int, batch: int) -> float:
    return float(top_k * batch)


def unique_experts_statistical(n_experts: int, top_k: int, batch: int,
                               probs: Optional[List[float]] = None) -> float:
    """E[unique experts] when each user independently draws top_k experts.

    With per-expert selection probability p_e (probability expert e is in one
    user's top-k), E[U] = sum_e 1-(1-p_e)^B. Uniform default: p_e = k/E.
    Exact under independence across users; top-k draws within a user are
    without replacement, captured by sum(p_e) = k.
    """
    if probs is None:
        p = top_k / n_experts
        return n_experts * (1.0 - (1.0 - p) ** batch)
    assert abs(sum(probs) - top_k) < 1e-6, "per-expert probs must sum to top_k"
    return sum(1.0 - (1.0 - pe) ** batch for pe in probs)


def zipf_probs(n_experts: int, top_k: int, alpha: float) -> List[float]:
    """Skewed per-expert selection probabilities, Zipf(alpha), scaled to sum=k.
    Clipped at 1 (an expert cannot be selected more than once per user)."""
    raw = [1.0 / (i + 1) ** alpha for i in range(n_experts)]
    s = sum(raw)
    p = [top_k * r / s for r in raw]
    # clip and renormalize the tail
    for _ in range(10):
        over = sum(max(0.0, x - 1.0) for x in p)
        if over < 1e-9:
            break
        under_ix = [i for i, x in enumerate(p) if x < 1.0]
        add = over / len(under_ix)
        p = [min(1.0, x) for x in p]
        for i in under_ix:
            p[i] = min(1.0, p[i] + add)
    return p


def make_trace(n_layers: int, n_experts: int, top_k: int, n_tokens: int,
               alpha: float = 0.0, seed: int = 0) -> List[List[List[int]]]:
    """Synthetic routing trace: trace[token][layer] = list of expert ids.
    alpha=0 -> uniform; alpha>0 -> Zipf-hot experts (per layer permutation)."""
    rng = random.Random(seed)
    perms = [rng.sample(range(n_experts), n_experts) for _ in range(n_layers)]
    weights = [1.0 / (i + 1) ** alpha if alpha > 0 else 1.0 for i in range(n_experts)]
    trace = []
    for _ in range(n_tokens):
        tok = []
        for l in range(n_layers):
            w = [weights[perms[l].index(e)] if alpha > 0 else 1.0 for e in range(n_experts)] \
                if alpha > 0 else None
            if alpha > 0:
                ids = []
                pool = list(range(n_experts))
                wl = list(w)
                for _ in range(top_k):
                    tot = sum(wl)
                    r = rng.random() * tot
                    acc = 0.0
                    for j, x in enumerate(wl):
                        acc += x
                        if acc >= r:
                            ids.append(pool[j]); pool.pop(j); wl.pop(j); break
                tok.append(ids)
            else:
                tok.append(rng.sample(range(n_experts), top_k))
        trace.append(tok)
    return trace


def unique_experts_trace(trace, batch: int, n_layers: int) -> float:
    """Average unique experts per layer per step, batching consecutive tokens."""
    tot, steps = 0.0, 0
    for s in range(0, len(trace) - batch + 1, batch):
        for l in range(n_layers):
            u = set()
            for b in range(batch):
                u.update(trace[s + b][l])
            tot += len(u)
        steps += 1
    return tot / (steps * n_layers) if steps else 0.0


def trace_stats(trace, n_experts: int):
    """Expert distribution stats: entropy (bits), max/mean load imbalance."""
    from collections import Counter
    c = Counter()
    for tok in trace:
        for layer in tok:
            c.update(layer)
    n = sum(c.values())
    probs = [c[e] / n for e in range(n_experts) if c[e] > 0]
    H = -sum(p * math.log2(p) for p in probs)
    loads = [c[e] for e in range(n_experts)]
    return {"entropy_bits": H, "max_over_mean_load": max(loads) / (n / n_experts),
            "experts_hit": len(probs)}
