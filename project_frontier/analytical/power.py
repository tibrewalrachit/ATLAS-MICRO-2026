"""Power model (Part XV).

Per-subsystem energy per decode step; every constant carries provenance.
DRAM constants are labeled and swept in the uncertainty analysis:
  - dram_array_pj_bit: DRAM core array read energy. Literature for 3D/stacked
    DRAM arrays: ~0.8-2.5 pJ/bit (HBM-class ~2.5-4 incl. interface).
    base=1.2 (assumed; 3D short-wire arrays), low=0.8, high=2.5.
  - iface_3d_pj_bit: vertical 3D interface (TSV/hybrid bond) energy.
    base=0.25 pJ/bit (hybrid bonding literature 0.05-0.5), low=0.1, high=0.6.
  - sram_pj_bit: 0.08 pJ/bit access (7nm-class SRAM macro, assumed).
  - noc_pj_bit_mm + controller/static fractions from ATLAS chip YAML scale.
Compute dynamic energy from compute.ComputeOrg.watts_per_tflops at utilization.
"""
from dataclasses import dataclass
from compute import ORGS

@dataclass
class PowerParams:
    dram_array_pj_bit: float = 1.2
    iface_3d_pj_bit: float = 0.25
    sram_pj_bit: float = 0.08
    sram_traffic_amp: float = 2.0        # bytes through SRAM per DRAM byte (fill+drain)
    noc_pj_bit: float = 0.35             # per-hop-average on-chip transport (ATLAS techfile scale)
    noc_traffic_frac: float = 0.3        # fraction of DRAM bytes crossing NoC
    static_frac: float = 0.12            # leakage as fraction of dynamic (assumed)
    controller_w_per_tbps: float = 0.4   # W per TB/s of streaming (scheduling logic, assumed)
    d2d_pj_bit: float = 1.3              # scale-out interconnect (ATLAS system yaml)

RANGES = {  # low/base/high (Part XXXIV)
    "dram_array_pj_bit": (0.8, 1.2, 2.5),
    "iface_3d_pj_bit": (0.1, 0.25, 0.6),
    "sram_pj_bit": (0.05, 0.08, 0.15),
    "static_frac": (0.08, 0.12, 0.20),
}


def step_power(step, hw, pp: PowerParams = PowerParams()):
    """Energy per decode step (J) by subsystem + average power (W)."""
    org = ORGS[hw.org]
    bits = step.bytes_per_step * 8
    e_dram = bits * pp.dram_array_pj_bit * 1e-12
    e_iface = bits * pp.iface_3d_pj_bit * 1e-12
    e_sram = bits * pp.sram_traffic_amp * pp.sram_pj_bit * 1e-12
    e_noc = bits * pp.noc_traffic_frac * pp.noc_pj_bit * 1e-12
    flops_t = step.by_category  # per-category flops
    tot_flops = sum(c["flops"] for c in flops_t.values())
    # dynamic compute energy: W/TFLOPs at full rate => J = W_per_TFLOPS * (FLOPs/1e12)
    e_comp = org.watts_per_tflops * (tot_flops / 1e12)
    e_ctrl = pp.controller_w_per_tbps * (step.bytes_per_step / 1e12)  # W*s per TB moved
    e_dyn = e_dram + e_iface + e_sram + e_noc + e_comp + e_ctrl
    e_static = e_dyn * pp.static_frac
    total = e_dyn + e_static
    by = dict(dram_array=e_dram, iface_3d=e_iface, sram=e_sram, noc=e_noc,
              compute=e_comp, controller=e_ctrl, static=e_static)
    watts = total / step.latency_s
    epj_token = total / (step.aggregate_tps * step.latency_s)  # J per token
    return {"energy_step_J": total, "by_subsystem_J": by, "avg_power_W": watts,
            "energy_per_token_J": epj_token, "tokens_per_J": 1.0 / epj_token}
