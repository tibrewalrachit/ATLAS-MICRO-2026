"""Part XXII: prefill/decode disaggregation sweep -> CSV."""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "serving"))
from deepseek_v4_flash.dag import kv_capacity_per_user as v4_kv
from minimax_m3.dag import kv_capacity_per_user as m3_kv
from disaggregation import request_time, INTERCONNECT_GRID_GBPS, LATENCY_GRID_S

PREFILL_TPS_GPU = 55000.0  # EXTERNAL ESTIMATE: 8xB200-class node on 13-23B-active MoE (labeled)

def main():
    rows = []
    for model, kvf, dstep in [("v4_flash", v4_kv, 0.00040),
                              ("m3_fp8", lambda c: m3_kv(c, 1.0), 0.00075)]:
        for ctx in (32768, 131072, 1048576):
            kv = kvf(ctx)
            for bw in INTERCONNECT_GRID_GBPS:
                for lat in LATENCY_GRID_S:
                    for n_out in (100, 800):
                        r = request_time(ctx, PREFILL_TPS_GPU, kv, bw, lat, n_out, dstep)
                        rows.append(dict(model=model, context=ctx,
                                         interconnect_GBps=bw, link_latency_us=lat * 1e6,
                                         n_output=n_out, kv_GB=round(kv / 1e9, 2),
                                         TTFT_s=round(r["TTFT_s"], 4),
                                         handoff_s=round(r["T_handoff_s"], 5),
                                         handoff_frac=round(r["handoff_frac_of_request"], 5)))
    out = os.path.join(HERE, "..", "results", "processed", "disaggregation_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    worst = max(rows, key=lambda r: r["handoff_frac"])
    print("worst handoff fraction:", worst)
    print("wrote", out, len(rows))

if __name__ == "__main__":
    main()
