"""Part XXI: serving-level experiments on the flagship configs."""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "serving"))
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from minimax_m3.dag import build_decode_dag as m3_dag
from model import FabrikPoint
from serving_sim import ServingSim

CTX_MIXES = {
    "short": lambda rng: rng.choice([8192, 16384, 32768]),
    "mixed": lambda rng: rng.choice([16384, 32768, 65536, 131072]),
    "long": lambda rng: rng.choice([131072, 262144, 524288]),
}

def main():
    rows = []
    cases = [("v4_flagship", lambda c: v4_dag(c), FabrikPoint("f50", 50, 1.0, 256, 64)),
             ("m3_flagship", lambda c: m3_dag(c, "exact_fp8"),
              FabrikPoint("f100", 100, 2.0, 512, 128))]
    for name, wb, hw in cases:
        sim = ServingSim(wb, hw, batch_cap=4)
        for mix, dist in CTX_MIXES.items():
            for rate in (1.0, 3.0, 6.0, 10.0):
                r = sim.run(rate, mean_output_tokens=600, context_dist=dist,
                            sim_seconds=150, seed=11)
                rows.append(dict(config=name, ctx_mix=mix, arrival_per_s=rate,
                                 **{k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in r.items()}))
                print(rows[-1], flush=True)
    out = os.path.join(HERE, "..", "results", "processed", "serving_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)

if __name__ == "__main__":
    main()
