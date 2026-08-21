"""Part XXVII: Pareto-frontier extraction from frontier.csv.

Objectives: maximize TPS_per_user, tokens_per_J; minimize cost_per_1m,
total_power_W, hardware proxy (BW+compute as cost drivers, via
cost_per_1m which already embeds hardware cost). Non-dominated filter per
(model, mode, context, batch); plus TPS-constrained frontiers.
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def dominates(a, b):
    ge = (a["TPS_per_user"] >= b["TPS_per_user"] and a["tokens_per_J"] >= b["tokens_per_J"]
          and a["cost_per_1m_output_tokens"] <= b["cost_per_1m_output_tokens"]
          and a["total_power_W"] <= b["total_power_W"])
    gt = (a["TPS_per_user"] > b["TPS_per_user"] or a["tokens_per_J"] > b["tokens_per_J"]
          or a["cost_per_1m_output_tokens"] < b["cost_per_1m_output_tokens"]
          or a["total_power_W"] < b["total_power_W"])
    return ge and gt


def pareto(rows):
    out = []
    for i, a in enumerate(rows):
        if not any(dominates(b, a) for b in rows if b is not a):
            out.append(a)
    return out


def main():
    rows = []
    for r in csv.DictReader(open(os.path.join(RES, "frontier.csv"))):
        if r["fits_capacity"] != "True" or r["feasible_capacity_mapping"] != "True":
            continue
        for k in ("TPS_per_user", "tokens_per_J", "cost_per_1m_output_tokens",
                  "total_power_W"):
            r[k] = float(r[k])
        rows.append(r)
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["model_mode"], r["context_tokens"], r["batch"]), []).append(r)
    fields = list(rows[0].keys())
    os.makedirs(os.path.join(RES, "pareto"), exist_ok=True)
    all_pareto = []
    for key, g in groups.items():
        pts = pareto(g)
        # dedupe identical objective tuples, keep min (BW, compute, sram) hw
        seen = {}
        for r in sorted(pts, key=lambda r: (float(r["requested_bw_TBps"]),
                                            float(r["peak_compute_TFLOPS"]),
                                            float(r["sram_MB"]))):
            k = (key, r["TPS_per_user"], r["tokens_per_J"],
                 r["cost_per_1m_output_tokens"], r["total_power_W"])
            if k not in seen:
                seen[k] = r
        all_pareto += list(seen.values())
    def write(name, rs):
        with open(os.path.join(RES, "pareto", name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rs)
        print(f"{name}: {len(rs)} rows")
    write("pareto_all.csv", all_pareto)
    for t in (1000, 1500, 2000):
        sub = [r for r in all_pareto if r["TPS_per_user"] >= t]
        write(f"pareto_{t}tps.csv", sub)


if __name__ == "__main__":
    main()
