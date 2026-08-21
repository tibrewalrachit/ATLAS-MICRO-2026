"""Part XVII: minimum hardware to hit each interactivity target."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

def main():
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "frontier.csv")))
            if r["fits_capacity"] == "True" and r["feasible_capacity_mapping"] == "True"]
    out = []
    keys = sorted({(r["model"], r["model_mode"], int(r["context_tokens"]), int(r["batch"]))
                   for r in rows})
    for model, mode, ctx, B in keys:
        g = [r for r in rows if r["model"] == model and r["model_mode"] == mode
             and int(r["context_tokens"]) == ctx and int(r["batch"]) == B]
        for target in (1000, 1500, 2000):
            ok = [r for r in g if float(r["TPS_per_user"]) >= target]
            if not ok:
                out.append(dict(model=model, mode=mode, context=ctx, batch=B,
                                target_tps=target, feasible=False))
                continue
            # min bandwidth first (the driving resource), then min compute/sram/power at that BW
            minbw = min(float(r["requested_bw_TBps"]) for r in ok)
            at_bw = [r for r in ok if float(r["requested_bw_TBps"]) == minbw]
            best = min(at_bw, key=lambda r: (float(r["peak_compute_TFLOPS"]),
                                             float(r["sram_MB"]), float(r["total_power_W"])))
            out.append(dict(model=model, mode=mode, context=ctx, batch=B,
                            target_tps=target, feasible=True,
                            min_bw_TBps=minbw,
                            min_compute_TFLOPS=float(best["peak_compute_TFLOPS"]),
                            min_sram_MB=float(best["sram_MB"]),
                            capacity_GB=float(best["memory_capacity_GB"]),
                            power_W=float(best["total_power_W"]),
                            tokens_per_J=float(best["tokens_per_J"]),
                            cost_per_1m=float(best["cost_per_1m_output_tokens"])))
    p = os.path.join(RES, "processed", "min_hardware_for_targets.csv")
    with open(p, "w", newline="") as f:
        fields = sorted({k for r in out for k in r}, key=lambda x: x != "model")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print("wrote", p, len(out), "rows")

if __name__ == "__main__":
    main()
