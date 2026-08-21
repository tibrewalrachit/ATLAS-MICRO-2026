"""Part XXXIV: Monte-Carlo uncertainty analysis.

Uncertain variables (low/base/high triangular): memory efficiencies (per
pattern), compute org utilization scale, DRAM array pJ/bit, 3D-interface
pJ/bit, static fraction, hardware cost, fleet utilization. For the flagship
configuration (50 TB/s, 1 PFLOP/s, 256 GB, 64 MB, hybrid) x V4-Flash @128K:
distribution of TPS/user, tokens/J, $/1M; plus one-at-a-time sensitivity
ranking (top-5 drivers of $/1M and TPS).
"""
import csv, os, random, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from model import FabrikPoint, decode_step
from power import step_power, PowerParams
from economics import cost_per_1m_tokens, TCOParams
from compute import ORGS

VARS = {  # name: (low, base, high)
    "eff_seq": (0.55, 0.801, 0.90),
    "eff_block": (0.55, 0.769, 0.85),
    "eff_random": (0.15, 0.287, 0.45),
    "eff_scan": (0.60, 0.809, 0.90),
    "compute_util_scale": (0.7, 1.0, 1.1),
    "dram_pj_bit": (0.8, 1.2, 2.5),
    "iface_pj_bit": (0.1, 0.25, 0.6),
    "static_frac": (0.08, 0.12, 0.20),
    "hardware_cost": (15000, 25000, 60000),
    "fleet_util": (0.4, 0.6, 0.8),
}


def tri(rng, lo, base, hi):
    return rng.triangular(lo, hi, base)


def evaluate(sample, wl):
    hw = FabrikPoint("mc", 50, 1.0, 256, 64, org="hybrid",
                     mem_eff={"seq_stream": sample["eff_seq"],
                              "block_gather": sample["eff_block"],
                              "random_gather": sample["eff_random"],
                              "scan": sample["eff_scan"]})
    org = ORGS["hybrid"]
    old = dict(org.gemv_util), org.attn_util, org.index_util
    for k in org.gemv_util:
        org.gemv_util[k] = min(0.98, org.gemv_util[k] * sample["compute_util_scale"])
    org.attn_util = min(0.98, old[1] * sample["compute_util_scale"])
    org.index_util = min(0.98, old[2] * sample["compute_util_scale"])
    try:
        r = decode_step(wl, hw, 1)
        p = step_power(r, hw, PowerParams(dram_array_pj_bit=sample["dram_pj_bit"],
                                          iface_3d_pj_bit=sample["iface_pj_bit"],
                                          static_frac=sample["static_frac"]))
        tco = cost_per_1m_tokens(r.aggregate_tps, TCOParams(
            hardware_cost_usd=sample["hardware_cost"],
            system_power_kw=p["avg_power_W"] / 1000,
            fleet_utilization=sample["fleet_util"]))
        return dict(tps=r.tps_per_user, tpj=p["tokens_per_J"],
                    cost=tco["cost_per_1m_output_tokens"], watts=p["avg_power_W"])
    finally:
        org.gemv_util.update(old[0]); org.attn_util = old[1]; org.index_util = old[2]


def main(n=2000, seed=7):
    rng = random.Random(seed)
    wl = v4_dag(131072)
    base = {k: v[1] for k, v in VARS.items()}
    base_r = evaluate(base, wl)
    samples = []
    for _ in range(n):
        s = {k: tri(rng, *v) for k, v in VARS.items()}
        samples.append(evaluate(s, wl))
    def pct(vals, p):
        vals = sorted(vals); return vals[int(p / 100 * (len(vals) - 1))]
    summary = {}
    for m in ("tps", "tpj", "cost", "watts"):
        vals = [s[m] for s in samples]
        summary[m] = {"p5": round(pct(vals, 5), 3), "p50": round(pct(vals, 50), 3),
                      "p95": round(pct(vals, 95), 3), "base": round(base_r[m], 3)}
    # one-at-a-time sensitivity on cost & tps
    sens = []
    for k, (lo, b, hi) in VARS.items():
        r_lo = evaluate({**base, k: lo}, wl)
        r_hi = evaluate({**base, k: hi}, wl)
        sens.append({"var": k,
                     "cost_swing": round(abs(r_hi["cost"] - r_lo["cost"]), 4),
                     "tps_swing": round(abs(r_hi["tps"] - r_lo["tps"]), 1)})
    sens.sort(key=lambda x: -x["cost_swing"])
    out = {"config": "fabrik 50TB/s 1PF 256GB 64MB hybrid, V4-Flash@128K B=1",
           "n_samples": n, "summary": summary,
           "sensitivity_by_cost_swing": sens,
           "top5_assumptions": [s["var"] for s in sens[:5]]}
    p = os.path.join(HERE, "..", "results", "processed", "uncertainty_mc.json")
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
