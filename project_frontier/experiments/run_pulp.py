"""PULP-on-CUBE analytical experiments (Phase 4) -> frontier_pulp.csv.

Workloads: qwen3_30b_a3b (official config), v4flash proxy at kv_read_fraction
{1.0, 0.25, 0.1} (exact V4 DAG scaled), plus the exact V4 DAG for reference.
Cross-checked against the Tensix-on-CUBE numbers (same S/M/L points).
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "architectures", "fabrik"))
sys.path.insert(0, os.path.join(HERE, "..", "architectures", "fabrik", "pulp"))
from generic.dag import build_decode_dag as gen_dag
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from model import decode_step, MOE_SHAPE
from workload_common import BYTES
from pulp_tile import make_pulp
from cube import make_v1

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MOE_SHAPE["qwen3_30b_a3b"] = (128, 8)


def qwen30(ctx):
    w = gen_dag(os.path.join(ROOT, "configs/models/qwen3_30b_a3b.json"), ctx,
                element_size=BYTES["fp8"], tp=1, ep=1, name="qwen3_30b_a3b",
                expert_element_size=BYTES["fp4"])
    w.weight_capacity_bytes = 21e9      # ~30.5B params, fp4 experts + fp8 rest
    w.kv_per_user_bytes = 48 * ctx * 4 * 128 * 2 * 1.0
    return w


def v4_proxy(ctx, frac):
    """Exact V4 DAG with attention spans scaled: labeled proxy for the
    kv_read_fraction sweep (indexer/compressor terms retained)."""
    w = v4_dag(int(ctx))
    if frac < 1.0:
        for op in w.ops:
            if op.category in ("csa_attn", "hca_attn", "local_attn"):
                op.kv_read_bytes *= frac
                op.flops *= frac
    w.mode = f"proxy_kvfrac{frac}"
    return w


def main():
    rows = []
    workloads = [("qwen3_30b_a3b", lambda c: qwen30(c))] + \
        [(f"v4flash_kvfrac{f}", (lambda f=f: lambda c: v4_proxy(c, f))())
         for f in (1.0, 0.25, 0.1)]
    for wname, wf in workloads:
        for size in ("S", "M", "L"):
            for hw_name, hw in [("pulp_default", make_pulp(size, "default")),
                                ("pulp_scaled", make_pulp(size, "scaled")),
                                ("tensix", make_v1(size, "tensix_tinytile"))]:
                for B in (1, 2, 4):
                    for ctx in (8192, 32768, 131072):
                        wl = wf(ctx)
                        if "v4flash" in wname:
                            MOE_SHAPE[wl.model] = (256, 6)
                        r = decode_step(wl, hw, B)
                        rows.append(dict(design=f"{hw_name}_{size}", workload=wname,
                                         B=B, ctx=ctx,
                                         step_ms=round(r.latency_s * 1e3, 4),
                                         tps_user=round(r.tps_per_user, 0),
                                         aggregate_tps=round(r.aggregate_tps, 0),
                                         dram_util=round(r.dram_util, 3),
                                         mat_util=round(r.compute_util, 3),
                                         noc_util=None, fits=r.fits_capacity,
                                         method="analytical"))
    out = os.path.join(HERE, "..", "results", "frontier_pulp.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", out, len(rows), "rows")
    # headline: PULP vs Tensix at S, B=1
    for wname in ("qwen3_30b_a3b", "v4flash_kvfrac1.0"):
        for hw_name in ("pulp_default", "pulp_scaled", "tensix"):
            r = next(x for x in rows if x["design"] == f"{hw_name}_S" and
                     x["workload"] == wname and x["B"] == 1 and x["ctx"] == 8192)
            print(f"{wname:20} {hw_name:13} S B=1 8K: {r['tps_user']:>7} tok/s "
                  f"dram={r['dram_util']:.2f} mat={r['mat_util']:.2f}")

if __name__ == "__main__":
    main()
