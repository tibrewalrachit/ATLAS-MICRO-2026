"""Memory-tier economics (FabrikSim §2, §5).

$/GB capacity and $/(TB/s) bandwidth per tier, the SRAM-resident capex floor
for routing-data-dependent MoE, and $ per (tok/s) of decode throughput for a
given workload. All vendor-derived figures are labeled estimates from public
sources; the 3D-DRAM column is the central uncertainty (low/base/high).
"""

TIERS = {  # $/GB capacity, $/(TB/s) bandwidth-slice, note
    "sram_wafer": {"usd_per_gb": 61000, "usd_per_tbps": 155,
                   "src": "CS-3 44GB/wafer, Groq LPU 230MB/chip; public price estimates"},
    "hbm3e_gpu": {"usd_per_gb": 156, "usd_per_tbps": 5600,
                  "src": "B300-class card cost over 288GB / 8TB/s (estimate)"},
    "3d_dram": {"usd_per_gb": (200, 400, 600), "usd_per_tbps": (300, 650, 1000),
                "src": "hybrid-bonded stack estimate — central study uncertainty"},
    "lpddr5x": {"usd_per_gb": 6, "usd_per_tbps": 78000 / 6.4 * 1,
                "src": "commodity; ~0.55 TB/s per 64GB package"},
    "nand": {"usd_per_gb": 0.10, "usd_per_tbps": None,
             "src": "in-flash compute ceiling ~7 tok/s for 671B (throughput-bound)"},
}


def sram_resident_capex_floor(model_bytes_gb, usd_per_gb=61000,
                              unit_gb=None, unit_usd=None):
    """Routing is data-dependent -> a resident architecture holds ALL weights.
    With unit granularity (e.g. CS-3 wafer: 44 GB / ~$3.5M) the floor rounds
    up to whole units: ceil(671/44)=16 wafers ~= $56M."""
    if unit_gb and unit_usd:
        import math
        return math.ceil(model_bytes_gb / unit_gb) * unit_usd
    return model_bytes_gb * usd_per_gb


def usd_per_tok_per_s(card_cost_usd, aggregate_tps):
    return card_cost_usd / aggregate_tps


def max_card_cost_to_match_gpu(gpu_cost_per_1m, aggregate_tps, tco_params):
    """Largest hardware cost at which Fabrik's $/1M <= the GPU reference,
    holding the other TCO inputs fixed. Closed form from the TCO identity."""
    from economics import TCOParams
    p = tco_params
    tokens_h = 3600 * aggregate_tps * p.fleet_utilization
    power_h = p.system_power_kw * p.pue * p.electricity_usd_kwh
    budget_h = gpu_cost_per_1m * tokens_h / 1e6
    capex_h = budget_h - power_h - p.other_hourly_usd
    return max(0.0, capex_h * 8760 * p.lifetime_years)
