"""Part XXIV: custom-silicon NRE amortization vs buying external accelerators."""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from model import FabrikPoint, decode_step
from power import step_power
from economics import cost_per_1m_tokens, TCOParams, effective_card_cost

NRE_GRID = [50e6, 100e6, 200e6, 400e6]
FLEET_GRID = [1000, 5000, 10000, 25000, 50000, 100000]
MFG_COST = 12000.0            # manufactured card+system share (assumed; swept in TCO)
GPU_REFERENCE_COST_PER_1M = 0.168   # cheapest same-model GPU route (baselines/external.csv)

def main():
    hw = FabrikPoint("flagship", 50, 1.0, 256, 64, org="hybrid")
    r = decode_step(v4_dag(131072), hw, 4)   # served at B=4 for economics
    p = step_power(r, hw)
    rows = []
    for nre in NRE_GRID:
        for fleet in FLEET_GRID:
            cost = effective_card_cost(MFG_COST, nre, fleet)
            tco = cost_per_1m_tokens(r.aggregate_tps, TCOParams(
                hardware_cost_usd=cost, system_power_kw=p["avg_power_W"] / 1000,
                fleet_utilization=0.6))
            rows.append(dict(nre_usd=int(nre), fleet_units=fleet,
                             effective_card_cost=round(cost, 0),
                             cost_per_1m=round(tco["cost_per_1m_output_tokens"], 4),
                             beats_gpu_reference=tco["cost_per_1m_output_tokens"]
                                                 < GPU_REFERENCE_COST_PER_1M))
    out = os.path.join(HERE, "..", "results", "processed", "nre_amortization.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r_ in rows:
        if r_["beats_gpu_reference"]:
            print(f"NRE ${r_['nre_usd']/1e6:.0f}M amortizes at {r_['fleet_units']} units "
                  f"(${r_['cost_per_1m']}/1M <= ${GPU_REFERENCE_COST_PER_1M})")
            break
    print("wrote", out)

if __name__ == "__main__":
    main()
