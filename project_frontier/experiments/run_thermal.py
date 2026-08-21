"""Part XVI: PyTA/HotSpot thermal evaluation of representative Fabrik points.

For each (design point, power envelope): build a floorplan from the physical
mapping (cores from compute, area from org area/TFLOPs + SRAM at 0.55 mm2/MB
[7nm-class macro, assumed] + 20% overhead), split power into logic vs DRAM
from the analytical power model at the workload operating point, run HotSpot
(liquid cooling, 85C threshold, 3D stack with num_memory_layer DRAM layers),
and record max temperature + thermally sustainable frequency.
"""
import os, sys, csv, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "architectures", "fabrik"))

from pyta.evaluator import thermal_evaluator
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from model import decode_step, FabrikPoint
from power import step_power
from compute import ORGS
from design_space import physicalize, make_point

SRAM_MM2_PER_MB = 0.55        # assumed 7nm-class
AREA_OVERHEAD = 1.20
NUM_DRAM_LAYERS = 8           # Fabrik: taller 3D stack than upstream's 4 (assumed)
COOLING = "liquid"
T_THRESH = 85.0

REP_POINTS = [   # (bw, pflops, cap, sram) — memory-bound, knee, compute-rich, hi-BW
    (24, 0.5, 256, 64), (40, 1.0, 256, 64), (50, 1.0, 256, 64),
    (60, 2.0, 384, 128), (100, 2.0, 384, 128), (150, 4.0, 512, 256),
]
ENVELOPES_W = [400, 500, 600, 800, 1000]


def evaluate(bw, pf, cap, sram, out_root):
    hw = make_point(bw, pf, cap, sram, "hybrid")
    phys = physicalize(bw, pf, cap, sram)
    org = ORGS["hybrid"]
    r = decode_step(v4_dag(131072), hw, 4)
    p = step_power(r, hw)
    by = p["by_subsystem_J"]; lat = r.latency_s
    logic_w = (by["compute"] + by["sram"] + by["noc"] + by["controller"] + by["static"]) / lat
    dram_w = (by["dram_array"] + by["iface_3d"]) / lat
    total_w = p["avg_power_W"]

    cores = phys["cores"]
    side = max(2, int(math.ceil(math.sqrt(cores))))
    comp_area = org.area_per_tflops * pf * 1000
    sram_area = SRAM_MM2_PER_MB * sram
    total_area_mm2 = (comp_area + sram_area) * AREA_OVERHEAD
    tile_mm = math.sqrt(total_area_mm2 / (side * side))
    rows = []
    for env in ENVELOPES_W:
        scale = min(1.0, env / total_w)   # power-capped operating point
        name = f"bw{bw}_c{pf}_s{sram}_env{env}"
        res = thermal_evaluator(
            name=name,
            tile_array_size=(side, side),
            tile_shape=(tile_mm * 1e-3, tile_mm * 1e-3),
            num_memory_layer=NUM_DRAM_LAYERS,
            cooling_style=COOLING,
            interposer_size=side * tile_mm * 1e-3 * 1.05,
            base_frequency=1000,
            # PyTA applies power PER TILE (see evaluator.run_atlas_hotspot_simulation)
            logic_power=logic_w * scale / (side * side),
            dram_power=dram_w * scale / (side * side),
            temperature_threshold=T_THRESH,
            # dict values replace LOGIC power per tile only (evaluator keeps dram_power)
            frequency_power_dict={f: logic_w * scale * (f / 1000) / (side * side)
                                  for f in (1000, 900, 800, 700, 600, 500, 400)},
            output_dir=os.path.join(out_root, name),
        ) or {}
        rows.append({
            "bw_TBps": bw, "pflops": pf, "capacity_GB": cap, "sram_MB": sram,
            "envelope_W": env, "unconstrained_W": round(total_w, 1),
            "power_scale": round(scale, 3),
            "logic_W": round(logic_w * scale, 1), "dram_W": round(dram_w * scale, 1),
            "die_mm": round(side * tile_mm, 1), "area_mm2": round(total_area_mm2, 1),
            "max_temperature_C": res.get("max_temperature_c"),
            "sustainable_frequency_MHz": res.get("frequency_MHz"),
            "thermally_valid": (res.get("max_temperature_c") or 999) <= T_THRESH,
        })
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    out_root = os.path.join(HERE, "..", "results", "raw", "thermal")
    os.makedirs(out_root, exist_ok=True)
    allrows = []
    pts = REP_POINTS if len(sys.argv) < 2 else [REP_POINTS[int(sys.argv[1])]]
    for bw, pf, cap, sram in pts:
        allrows += evaluate(bw, pf, cap, sram, out_root)
    out = os.path.join(HERE, "..", "results", "processed", "thermal_summary.csv")
    with open(out, "a" if len(sys.argv) >= 2 else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        if f.tell() == 0:
            w.writeheader()
        w.writerows(allrows)
    print("wrote", out)
