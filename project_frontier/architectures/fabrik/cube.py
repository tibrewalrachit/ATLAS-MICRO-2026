"""FabrikV1: Tensix logic die on Winbond CUBE-class 3D DRAM stacks.

Per user-supplied Polaris/tt_bh study (cross-validated here):
  - CUBE stack: 1024 I/O x 2 GT/s = 256 GB/s, 4 GB per stack
  - Tensix core (tt_bh.yaml): 2048 FP8 MACs/clk @ 1 GHz = 4.096 TFLOP/s FP8
  - V1-S/M/L = 64/128/192 cores with 16/32/48 stacks
ATLAS cycle mapping: one ATLAS core = one CUBE stack (4 x 64GB/s channels,
cloud_0.25TBps Ramulator config) with mac_num = 4 Tensix cores' MACs (8192),
preserving the compute:memory ratio and per-stack channel structure; 16-core
chip = V1-S. NoC: 4x4 mesh.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analytical"))
from model import FabrikPoint

CUBE_STACK_TBPS = 0.256
CUBE_STACK_GB = 4.0
TENSIX_CORE_TFLOPS = 4.096      # 2048 MACs x 2 x 1 GHz

V1 = {
    "S": dict(cores=64, stacks=16),
    "M": dict(cores=128, stacks=32),
    "L": dict(cores=192, stacks=48),
}


def make_v1(size: str, org: str = "tensix_tinytile", sram_mb_per_core: float = 1.5):
    c = V1[size]
    return FabrikPoint(
        f"fabrikv1_{size}", peak_bw_TBps=c["stacks"] * CUBE_STACK_TBPS,
        peak_pflops_fp8=c["cores"] * TENSIX_CORE_TFLOPS / 1000,
        capacity_GB=c["stacks"] * CUBE_STACK_GB,
        sram_MB=c["cores"] * sram_mb_per_core, org=org)


ATLAS_CYCLE_CHIP = """architecture:
  frequency: 1000
  core_num: 16
  core:
    controller: {power: 1.0, area: 1.0}
    matrix: {mac_num: 8192, power: 1.1, area: 1.1}
    vector: {vec_num: 512, power: 1.2, area: 1.2}
    buffer: {buffer_size: 6144, read_bw: 8192, write_bw: 8192, power: 1.3, area: 1.3}
  dram: {config_path: configs/architecture/dram/cloud/cloud_0.25TBps.yaml, power: 1.4, area: 1.4}
  noc: {topology: mesh, config_path: configs/architecture/noc/4x4_mesh, flit_size: 64, power: 1.5, area: 1.5}
"""
