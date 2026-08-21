"""Parts XIII-XIV: compute organization & specialized index-engine study."""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from minimax_m3.dag import build_decode_dag as m3_dag
from model import FabrikPoint, decode_step
from power import step_power
from compute import ORGS

def main():
    rows = []
    cases = [("v4_128k", lambda: v4_dag(131072), 50, 1.0, 256, 64),
             ("v4_1m", lambda: v4_dag(1048576), 50, 1.0, 256, 64),
             ("m3_128k", lambda: m3_dag(131072, "exact_fp8"), 100, 2.0, 512, 128),
             ("m3_1m", lambda: m3_dag(1048576, "exact_fp8"), 100, 2.0, 512, 128)]
    for cname, wf, bw, pf, cap, sram in cases:
        for org in ORGS:
            for B in (1, 4):
                hw = FabrikPoint(f"{cname}_{org}", bw, pf, cap, sram, org=org)
                r = decode_step(wf(), hw, B)
                p = step_power(r, hw)
                o = ORGS[org]
                rows.append(dict(case=cname, org=org, batch=B,
                                 tps_user=round(r.tps_per_user, 1),
                                 tokens_per_J=round(p["tokens_per_J"], 2),
                                 compute_util=round(r.compute_util, 3),
                                 area_mm2_compute=round(o.area_per_tflops * pf * 1000, 0),
                                 watts=round(p["avg_power_W"], 0)))
    out = os.path.join(HERE, "..", "results", "processed", "compute_orgs.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # headline: hybrid (with index engine) vs others at 1M B=1
    for cname in ("v4_1m", "m3_1m"):
        base = next(r for r in rows if r["case"] == cname and r["org"] == "atlas_matrix" and r["batch"] == 1)
        for org in ("systolic_fixed", "reconfig_systolic", "gemv_lanes", "hybrid"):
            x = next(r for r in rows if r["case"] == cname and r["org"] == org and r["batch"] == 1)
            print(f"{cname} B=1 {org:18}: {x['tps_user']:7.1f} TPS "
                  f"({x['tps_user']/base['tps_user']:.2f}x vs atlas_matrix), "
                  f"{x['tokens_per_J']} tok/J")
    print("wrote", out)

if __name__ == "__main__":
    main()
