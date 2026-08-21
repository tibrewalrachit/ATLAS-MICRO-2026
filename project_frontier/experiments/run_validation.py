"""Part XIX: analytical model vs ATLAS cycle-level end-to-end validation.

Compares decode-step latency predicted by the analytical framework against
cycle-simulated e2e results for ATLAS-supported models on the stock cloud
chip (16 cores x 1.024 TB/s vaults = 16.4 TB/s, 262 TFLOP/s fp16, 48 MB SRAM).
"""
import csv, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
from generic.dag import build_decode_dag
from model import FabrikPoint, decode_step, MOE_SHAPE

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# stock cloud chip as a FabrikPoint: element_size=2 runs are fp16-class MACs
TEST_CHIP = FabrikPoint("atlas_test_chip", peak_bw_TBps=16.384,
                        peak_pflops_fp8=0.262, capacity_GB=128, sram_MB=48,
                        org="atlas_matrix")

CASES = [  # (name, config, ctx, batch, tp, ep, cycle_csv_glob)
    ("opt_66b", "configs/models/opt_66b.json", 4096, 8, 8, 8,
     "project_frontier/results/ub/upstream_baseline/auto/**/opt_66b*/e2e_performance.csv"),
    ("mixtral_8x22b", "configs/models/mixtral_8x22b.json", 16384, 8, 8, 8,
     "project_frontier/results/ub/upstream_baseline/atlang/**/mixtral*/e2e_performance.csv"),
]


def main():
    rows = []
    for name, cfgp, ctx, bs, tp, ep, pat in CASES:
        wl = build_decode_dag(os.path.join(ROOT, cfgp), ctx, 2, tp=tp, name=name)
        MOE_SHAPE[wl.model] = getattr(wl, "moe_shape", (1, 1))
        r = decode_step(wl, TEST_CHIP, bs, "statistical")
        matches = glob.glob(os.path.join(ROOT, pat), recursive=True)
        cyc = None
        if matches:
            row = list(csv.DictReader(open(matches[0])))[0]
            cyc = float(row["e2e_latency_s"])
        ratio = (r.latency_s / cyc) if cyc else None
        rows.append(dict(case=name, ctx=ctx, batch=bs,
                         analytical_ms=round(r.latency_s * 1e3, 3),
                         cycle_ms=round(cyc * 1e3, 3) if cyc else None,
                         ratio=round(ratio, 2) if ratio else None))
        print(rows[-1])
    out = os.path.join(HERE, "..", "results", "processed", "validation_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
