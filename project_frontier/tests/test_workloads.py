"""Validation gates: DAG builders vs official figures and closed forms."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))

from deepseek_v4_flash.dag import build_decode_dag as v4_dag, total_params as v4_params
from minimax_m3.dag import build_decode_dag as m3_dag, total_params as m3_params
import attention as cf


def close(a, b, tol):
    assert abs(a - b) / max(abs(b), 1e-30) < tol, f"{a} vs {b} (tol {tol})"


def test_v4_totals():
    p = v4_params()
    close(p["total"], 284e9, 0.02)                      # official 284B
    active = (p["total"] - p["routed"]) + 6/256 * p["routed"] \
             - p["embed_head"] / 2                       # embed row lookup ~0
    assert 12.5e9 < active < 14.5e9                      # official ~13B


def test_m3_totals():
    p = m3_params()
    close(p["total"], 428e9, 0.02)                       # official 428B (MTP excluded here)


def test_v4_kv_index_closed_form():
    for ctx in (32768, 131072, 1048576):
        t = v4_dag(ctx).totals()
        kv_dag = sum(c["kv_r"] for c in t.values())
        idx_dag = sum(c["idx"] for c in t.values())
        close(kv_dag, cf.v4_kv_read_bytes_per_token(ctx), 0.02)
        close(idx_dag, cf.v4_index_bytes_per_token(ctx), 0.02)


def test_m3_kv_index_closed_form():
    for ctx in (32768, 131072, 1048576):
        t = m3_dag(ctx, "exact_fp8").totals()
        kv_dag = sum(c["kv_r"] for c in t.values())
        idx_dag = sum(c["idx"] for c in t.values())
        close(kv_dag, cf.m3_kv_read_bytes_per_token(ctx), 0.03)
        close(idx_dag, cf.m3_index_bytes_per_token(ctx), 0.03)


def test_moe_reuse_monotone():
    sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
    import moe
    prev = 0
    for B in (1, 2, 4, 8):
        u = moe.unique_experts_statistical(256, 6, B)
        assert u > prev and u <= 6 * B
        prev = u


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")


def test_v3_totals_and_mla():
    """FabrikSim cross-validation: independently-written V3 DAG must reproduce
    the session-summary numbers from the official config."""
    from deepseek_v3.dag import build_decode_dag, total_params, MLA_FLOP_PER_BYTE
    p = total_params()
    close(p["total"], 671e9, 0.005)
    close(MLA_FLOP_PER_BYTE, 484, 0.01)
    s = build_decode_dag(8192).summary()
    close(s["bytes"], 36.9e9, 0.02)          # FabrikSim: 36.9 GB/step B=1 8K
    t = build_decode_dag(131072).totals()
    attn_frac = t["full_attn"]["flops"] / sum(c["flops"] for c in t.values())
    assert attn_frac > 0.95                   # FabrikSim: 97% of FLOPs @128K
