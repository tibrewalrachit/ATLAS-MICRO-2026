"""FabrikV1 (Tensix + CUBE) analytical cross-check vs the Polaris/tt_bh study.

Differences from that study, on purpose: pattern-calibrated memory
efficiencies (Ramulator-measured) instead of flat 80%; Tensix tile-utilization
brackets instead of ideal compute; explicit MoE reuse model. Capacity check
includes weights + B x KV.
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "architectures", "fabrik"))
from gpt_oss_120b.dag import build_decode_dag as oss_dag, weight_capacity_gb as oss_cap, \
    kv_capacity_per_user as oss_kv
from generic.dag import build_decode_dag as gen_dag
from model import decode_step, MOE_SHAPE
from cube import make_v1, V1
from workload_common import BYTES

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MOE_SHAPE["gpt_oss_120b"] = (128, 4)
MOE_SHAPE["qwen3_235b"] = (128, 8)

def qwen_wl(ctx):
    w = gen_dag(os.path.join(ROOT, "configs/models/qwen3_235b_a22b.json"), ctx,
                element_size=BYTES["fp8"], tp=1, ep=1, name="qwen3_235b",
                expert_element_size=BYTES["fp4"])
    return w

def qwen_cap_gb():
    import json
    c = json.load(open(os.path.join(ROOT, "configs/models/qwen3_235b_a22b.json")))
    experts = c["num_hidden_layers"] * c["num_experts"] * 3 * c["hidden_size"] * c["moe_intermediate_size"]
    total = 235e9
    return (experts * BYTES["fp4"] + (total - experts) * BYTES["fp8"]) / 1e9

def main():
    rows = []
    user_ref = {("gpt_oss_120b", "S", 1, 8192): 942, ("gpt_oss_120b", "M", 1, 8192): 1883,
                ("gpt_oss_120b", "L", 1, 8192): 2825, ("gpt_oss_120b", "S", 1, 131072): 570,
                ("gpt_oss_120b", "M", 1, 131072): 1141, ("gpt_oss_120b", "L", 1, 131072): 1711,
                ("qwen3_235b", "M", 1, 8192): 431, ("qwen3_235b", "L", 1, 8192): 646,
                ("gpt_oss_120b", "S", 4, 8192): 365}
    for model, wlf, capf, kvf in [
            ("gpt_oss_120b", oss_dag, oss_cap, lambda c: oss_kv(c) / 1e9),
            ("qwen3_235b", qwen_wl, qwen_cap_gb, None)]:
        for size in ("S", "M", "L"):
            for org in ("tensix_tinytile", "tensix_tile32"):
                hw = make_v1(size, org)
                for B in (1, 4):
                    for ctx in (8192, 131072):
                        wl = wlf(ctx)
                        wl.weight_capacity_bytes = capf() * 1e9
                        wl.kv_per_user_bytes = (kvf(ctx) if kvf else 0.06 * ctx / 8192) * 1e9
                        r = decode_step(wl, hw, B)
                        fits = r.fits_capacity
                        ref = user_ref.get((model, size, B, ctx))
                        rows.append(dict(model=model, size=size, org=org, batch=B,
                                         ctx=ctx, bw_TBps=round(hw.peak_bw_TBps, 2),
                                         cap_GB=hw.capacity_GB,
                                         tps_user=round(r.tps_per_user, 0),
                                         compute_util=round(r.compute_util, 3),
                                         fits=fits, polaris_ref=ref))
    out = os.path.join(HERE, "..", "results", "processed", "fabrikv1_cube.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["polaris_ref"] or (r["org"] == "tensix_tinytile" and r["batch"] == 1):
            print(r)
    print("wrote", out)

if __name__ == "__main__":
    main()
