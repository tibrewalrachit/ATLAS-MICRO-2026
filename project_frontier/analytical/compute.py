"""Compute-organization model (Part XIII).

Four organizations, each defined by how efficiently it turns peak FLOP/s into
useful FLOP/s for the decode operator classes at batch M=1/2/4:

- atlas_matrix: ATLAS's MAC-throughput engine (shape-agnostic MAC pool). Its
  cycle model is ceil(MACs/mac_num); real utilization is set by SRAM/DRAM feed.
- systolic_fixed: classic 128x128 weight-stationary array. GEMV (M<=4) wastes
  rows: util ~ M/128 unless K-folding is applied; we model output-stationary
  with K-folding giving util = min(1, M*fold_factor/128), fold_factor=16
  (16 K-slices resident) -> M=1 util 12.5%.
- reconfig_systolic: reconfigurable aspect ratio (64x64 .. 2x2048): keeps all
  lanes busy for skinny M by widening N; util limited by output reduction
  overhead: util = base_util * (1 - reconfig_overhead).
- gemv_lanes: dedicated dot-product lanes (M=1..4 native): high util for GEMV
  but lower peak per area; attention scores/AV also run well (vector-like).
- hybrid: reconfig systolic for GEMM/attention + gemv lanes for expert GEMVs
  + index engine for scans (Part XIV).

Peak FLOP/s is FP8; FP4 doubles rate on orgs flagged fp4_double.
Area/power scalars are relative to atlas_matrix at iso-peak (from ATLAS
cloud chip YAML: matrix 7680 MACs @1GHz = 15.4 TFLOP/s costing 13.13 mm2 /
3.125 W per core; assumption_level=derived from ATLAS numbers).
"""
from dataclasses import dataclass

@dataclass
class ComputeOrg:
    name: str
    gemv_util: dict          # {1: u, 2: u, 4: u} for M=1/2/4 GEMV/GEMM-skinny
    attn_util: float         # decode attention (seq x head_dim GEMVs, MQA/GQA)
    index_util: float        # index scan / top-k throughput fraction
    fp4_double: bool
    area_per_tflops: float   # mm^2 per peak FP8 TFLOP/s
    watts_per_tflops: float  # W per peak FP8 TFLOP/s (dynamic, at full util)

ATLAS_AREA_PER_TFLOPS = 13.1255 / 15.36   # mm2 per TFLOP/s (fp16-class MACs)
ATLAS_W_PER_TFLOPS = 3.125 / 15.36

ORGS = {
    "atlas_matrix": ComputeOrg("atlas_matrix", {1: 0.70, 2: 0.75, 4: 0.80},
                               0.60, 0.30, False,
                               ATLAS_AREA_PER_TFLOPS, ATLAS_W_PER_TFLOPS),
    "systolic_fixed": ComputeOrg("systolic_fixed", {1: 0.125, 2: 0.25, 4: 0.50},
                                 0.45, 0.20, True,
                                 0.55 * ATLAS_AREA_PER_TFLOPS, 0.75 * ATLAS_W_PER_TFLOPS),
    "reconfig_systolic": ComputeOrg("reconfig_systolic", {1: 0.60, 2: 0.70, 4: 0.80},
                                    0.55, 0.30, True,
                                    0.70 * ATLAS_AREA_PER_TFLOPS, 0.85 * ATLAS_W_PER_TFLOPS),
    "gemv_lanes": ComputeOrg("gemv_lanes", {1: 0.90, 2: 0.90, 4: 0.90},
                             0.80, 0.50, True,
                             1.15 * ATLAS_AREA_PER_TFLOPS, 1.0 * ATLAS_W_PER_TFLOPS),
    "hybrid": ComputeOrg("hybrid", {1: 0.85, 2: 0.88, 4: 0.90},
                         0.80, 0.90, True,
                         0.95 * ATLAS_AREA_PER_TFLOPS, 0.95 * ATLAS_W_PER_TFLOPS),
}

def op_util(org: ComputeOrg, engine: str, category: str, batch: int) -> float:
    if engine == "index":
        return org.index_util
    if category in ("csa_attn", "hca_attn", "local_attn", "msa_attn", "full_attn"):
        return org.attn_util
    b = 1 if batch <= 1 else (2 if batch <= 2 else 4)
    return org.gemv_util[b]
