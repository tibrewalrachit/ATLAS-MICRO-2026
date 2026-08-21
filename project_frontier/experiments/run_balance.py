"""FabrikSim incorporation: balance ratios, per-operator bandwidth
provisioning, tier economics, SRAM capex floor, GPU contention bound.
Outputs results/processed/balance_summary.json + prints report-ready tables.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "economics"))

from deepseek_v3.dag import build_decode_dag as v3_dag, total_params as v3_params, \
    kv_capacity_per_user as v3_kv, MLA_FLOP_PER_BYTE
from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from model import MOE_SHAPE, FabrikPoint, decode_step
from balance import demanded_balance, operator_bandwidth_split, REF_MACHINES
from tiers import TIERS, sram_resident_capex_floor, max_card_cost_to_match_gpu
from economics import TCOParams

MOE_SHAPE["deepseek_v3"] = (256, 8)
out = {}

# --- 1. demanded balance table (GB per TB/s), V3 @8K ---
wl = v3_dag(8192)
wcap = v3_params()["total"] / 1e9          # fp8 GB
tbl = {}
for B in (1, 4, 16, 64):
    for T in (1000, 2000, 4000):
        r = demanded_balance(wl, wcap, v3_kv(8192) / 1e9, B, T)
        tbl[f"B{B}_T{T}"] = round(r["gb_per_tbps"], 2)
out["demanded_gb_per_tbps_v3_8k"] = tbl
out["card_ratio_128gb_105tbps"] = round(128 / 105, 2)

# --- 2. per-operator bandwidth provisioning, V3 @128K, 2 PFLOP/s ---
split = operator_bandwidth_split(v3_dag(131072), 2.0)
out["operator_split_v3_128k_2pf"] = {
    k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in split.items()}
out["mla_flop_per_byte"] = round(MLA_FLOP_PER_BYTE, 1)
out["machine_balance_flop_per_byte"] = {k: round(v, 0) for k, v in REF_MACHINES.items()}

# --- 3. SRAM capex floor & tier table ---
out["sram_capex_floor_671gb_usd_m"] = {
    "per_gb_basis": round(sram_resident_capex_floor(671) / 1e6, 1),
    "wafer_granularity": round(sram_resident_capex_floor(671, unit_gb=44, unit_usd=3.5e6) / 1e6, 1)}
out["tiers"] = TIERS

# --- 4. GPU contention bound (V4-Flash, public data) ---
# Roofline single-user TPS on an 8xB300-class group serving V4-Flash:
# 11.2 GB/token over 8x8 TB/s at ideal 0.8 efficiency, perfectly parallel.
roofline_tps = (8 * 8e12 * 0.8) / 11.2e9
measured_best = 306.0     # best measured provider (Artificial Analysis)
out["gpu_contention_bound_v4"] = {
    "roofline_tps_8xB300_ideal": round(roofline_tps, 0),
    "best_measured_provider_tps": measured_best,
    "contention_factor_lower_bound": round(roofline_tps / measured_best, 1),
    "note": "roofline assumes perfect TP/EP scaling and zero collectives; "
            "measured/ideal gap >= real contention. A cycle-accurate GPU "
            "number requires a GPU simulator (outside ATLAS scope) - this "
            "bound uses only public measurements + arithmetic."}

# --- 5. V3 on Fabrik flagships (fp8): does the V4 story transfer? ---
for name, bw, pf, cap, sram in [("f50", 50, 1.0, 683, 64), ("f105", 105, 2.0, 683, 128)]:
    hw = FabrikPoint(name, bw, pf, cap, sram, org="hybrid")
    res = {}
    for B in (1, 4):
        for ctx in (8192, 131072):
            r = decode_step(v3_dag(ctx), hw, B)
            res[f"B{B}_ctx{ctx}"] = {"tps_user": round(r.tps_per_user, 0),
                                     "fits": r.fits_capacity}
    out[f"v3_on_{name}"] = res

os.makedirs(os.path.join(HERE, "..", "results", "processed"), exist_ok=True)
p = os.path.join(HERE, "..", "results", "processed", "balance_summary.json")
json.dump(out, open(p, "w"), indent=2)
print(json.dumps(out, indent=2))
