"""Full analytical DSE -> results/frontier.csv (Parts XVII, XVIII, XXX)."""
import csv, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "architectures", "fabrik"))

from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from minimax_m3.dag import build_decode_dag as m3_dag
from model import decode_step
from power import step_power, PowerParams
from economics import cost_per_1m_tokens, TCOParams
from design_space import full_grid, physicalize, make_point, BW_GRID_TBPS, \
    COMPUTE_GRID_PFLOPS, CAPACITY_GRID_GB, SRAM_GRID_MB

CONTEXTS = [32768, 131072, 1048576]
BATCHES = [1, 2, 4]

FIELDS = ["model", "model_mode", "batch", "context_tokens", "memory_capacity_GB",
          "requested_bw_TBps", "achieved_bw_TBps", "dram_utilization",
          "peak_compute_TFLOPS", "achieved_compute_TFLOPS", "compute_utilization",
          "sram_MB", "org", "decode_step_ms", "TPS_per_user", "aggregate_TPS",
          "weight_bytes_per_token", "kv_bytes_per_token", "index_bytes_per_token",
          "total_bytes_per_token", "flops_per_token", "arithmetic_intensity",
          "logic_power_W", "dram_power_W", "interface_power_W", "total_power_W",
          "energy_per_token_J", "tokens_per_J", "useful_TFLOPS_per_W",
          "max_temperature_C", "sustainable_frequency_MHz",
          "hardware_cost_USD", "fleet_utilization", "cost_per_1m_output_tokens",
          "meets_1000_tps", "meets_1500_tps", "meets_2000_tps",
          "fits_capacity", "feasible_capacity_mapping",
          "simulation_method", "assumption_level", "reuse_model"]


def workload_cache():
    cache = {}
    for ctx in CONTEXTS:
        cache[("deepseek_v4_flash", "exact", ctx)] = v4_dag(ctx)
        cache[("minimax_m3", "exact_fp8", ctx)] = m3_dag(ctx, "exact_fp8")
        cache[("minimax_m3", "exact_bf16", ctx)] = m3_dag(ctx, "exact_bf16")
    return cache


def run(out_path, orgs=("hybrid",), reuse="statistical",
        cost_usd=25000.0, fleet_util=0.6, thermal_lookup=None, quiet=False):
    wls = workload_cache()
    rows = []
    for (mname, mode, ctx), wl in wls.items():
        for (hw, phys) in full_grid(orgs):
            for B in BATCHES:
                r = decode_step(wl, hw, B, reuse)
                p = step_power(r, hw, PowerParams())
                by = r.by_category
                w_b = sum(c["bytes"] for k, c in by.items()
                          if k in ("dense_proj", "routed_expert", "shared_expert",
                                   "router", "head", "residual_hc", "indexer",
                                   "csa_attn", "hca_attn")) / B  # incl. small proj weights
                kv_b = sum(c["bytes"] for k, c in by.items()
                           if k in ("full_attn", "msa_attn", "local_attn")) / B
                idx_b = sum(c["bytes"] for k, c in by.items()
                            if k in ("msa_index",)) / B
                tco = cost_per_1m_tokens(r.aggregate_tps, TCOParams(
                    hardware_cost_usd=cost_usd, system_power_kw=p["avg_power_W"] / 1000,
                    fleet_utilization=fleet_util))
                therm = thermal_lookup(hw, p["avg_power_W"]) if thermal_lookup else \
                        {"max_temperature_C": None, "sustainable_frequency_MHz": None}
                by_sub = p["by_subsystem_J"]
                logic_w = (by_sub["compute"] + by_sub["sram"] + by_sub["noc"]
                           + by_sub["controller"]) / r.latency_s
                rows.append(dict(
                    model=mname, model_mode=mode, batch=B, context_tokens=ctx,
                    memory_capacity_GB=hw.capacity_GB, requested_bw_TBps=hw.peak_bw_TBps,
                    achieved_bw_TBps=round(r.bytes_per_step / r.latency_s / 1e12, 3),
                    dram_utilization=round(r.dram_util, 4),
                    peak_compute_TFLOPS=hw.peak_pflops_fp8 * 1000,
                    achieved_compute_TFLOPS=round(sum(c["flops"] for c in by.values())
                                                  / r.latency_s / 1e12, 2),
                    compute_utilization=round(r.compute_util, 4),
                    sram_MB=hw.sram_MB, org=hw.org,
                    decode_step_ms=round(r.latency_s * 1e3, 4),
                    TPS_per_user=round(r.tps_per_user, 1),
                    aggregate_TPS=round(r.aggregate_tps, 1),
                    weight_bytes_per_token=round(w_b, 0),
                    kv_bytes_per_token=round(kv_b, 0),
                    index_bytes_per_token=round(idx_b, 0),
                    total_bytes_per_token=round(r.bytes_per_token, 0),
                    flops_per_token=round(r.flops_per_token, 0),
                    arithmetic_intensity=round(r.ai, 3),
                    logic_power_W=round(logic_w, 1),
                    dram_power_W=round(by_sub["dram_array"] / r.latency_s, 1),
                    interface_power_W=round(by_sub["iface_3d"] / r.latency_s, 1),
                    total_power_W=round(p["avg_power_W"], 1),
                    energy_per_token_J=round(p["energy_per_token_J"], 6),
                    tokens_per_J=round(p["tokens_per_J"], 2),
                    useful_TFLOPS_per_W=round(sum(c["flops"] for c in by.values())
                                              / r.latency_s / 1e12 / p["avg_power_W"], 4),
                    max_temperature_C=therm["max_temperature_C"],
                    sustainable_frequency_MHz=therm["sustainable_frequency_MHz"],
                    hardware_cost_USD=cost_usd, fleet_utilization=fleet_util,
                    cost_per_1m_output_tokens=round(tco["cost_per_1m_output_tokens"], 4),
                    meets_1000_tps=r.tps_per_user >= 1000,
                    meets_1500_tps=r.tps_per_user >= 1500,
                    meets_2000_tps=r.tps_per_user >= 2000,
                    fits_capacity=r.fits_capacity,
                    feasible_capacity_mapping=phys["feasible_capacity"],
                    simulation_method="analytical", assumption_level="assumed_defaults",
                    reuse_model=reuse))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=FIELDS)
        wcsv.writeheader(); wcsv.writerows(rows)
    if not quiet:
        print(f"wrote {len(rows)} rows -> {out_path}")
    return rows


if __name__ == "__main__":
    import time
    t0 = time.time()
    run(os.path.join(HERE, "..", "results", "frontier.csv"))
    print(f"DSE wall time: {time.time()-t0:.1f}s")
