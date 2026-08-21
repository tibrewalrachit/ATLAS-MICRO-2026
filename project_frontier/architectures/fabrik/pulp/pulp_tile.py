"""PULP tile (Snitch + RedMulE) on CUBE — ATLAS chip configs + FabrikPoints.

All parameters from params/pulp_tile.yaml (sourced). One ATLAS core = one
CUBE stack = 4 PULP tiles. Design points match FabrikV1 S/M/L (64/128/192
tiles over 16/32/48 stacks) for direct Tensix comparison.
Two matrix variants: redmule_default (4x48 CE = 192 MACs/stack) and
redmule_scaled (4x512 CE = 2048 MACs/stack; parametric, no silicon ref).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "analytical"))
from model import FabrikPoint

CUBE_STACK_TBPS = 0.256
CUBE_STACK_GB = 4.0
TILE_MATRIX_TFLOPS = {"default": 0.096, "scaled": 1.024,   # 48/512 CE x 2 x 1GHz
                      # hetero: NEUREKA 36PE int8-base 4608 MAC/clk = 9.2 TOPS-fp4eq
                      # counted at its 8-bit base rate (fp4_double in the org
                      # applies the bit-serial gain) + RedMulE 96 GFLOP fp16
                      "hetero": 4.608 * 2 / 1000 * 1000 / 1000 + 0.096}
V1 = {"S": 16, "M": 32, "L": 48}                            # stacks (=4 tiles each)


def make_pulp(size, variant="default", freq_ghz=1.0):
    stacks = V1[size]
    tiles = stacks * 4
    return FabrikPoint(
        f"pulp_{variant}_{size}", peak_bw_TBps=stacks * CUBE_STACK_TBPS,
        peak_pflops_fp8=tiles * TILE_MATRIX_TFLOPS[variant] * freq_ghz / 1000,
        capacity_GB=stacks * CUBE_STACK_GB,
        sram_MB=tiles * 0.128,
        org="pulp_hetero" if variant == "hetero" else "pulp_redmule")


CHIP_TEMPLATE = """architecture:
  frequency: {freq}
  core_num: 16
  core:
    controller: {{power: 1.0, area: 1.0}}
    matrix: {{mac_num: {macs}, power: 1.1, area: 1.1}}
    vector: {{vec_num: 128, power: 1.2, area: 1.2}}
    buffer: {{buffer_size: 512, read_bw: 1024, write_bw: 1024, power: 1.3, area: 1.3}}
  dram: {{config_path: configs/architecture/dram/cloud/cloud_0.25TBps.yaml, power: 1.4, area: 1.4}}
  noc: {{topology: mesh, config_path: configs/architecture/noc/4x4_mesh, flit_size: 64, power: 1.5, area: 1.5}}
"""

VARIANT_MACS = {"default": 192, "scaled": 2048,
                "hetero": 4 * (4608 + 48)}   # per-stack: 4 x (NEUREKA int8-base + RedMulE)


def emit(out_dir, freq_mhz=1000):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for v, macs in VARIANT_MACS.items():
        p = os.path.join(out_dir, f"pulp_tile_{v}.yaml")
        open(p, "w").write(CHIP_TEMPLATE.format(macs=macs, freq=freq_mhz))
        out[v] = p
    return out


if __name__ == "__main__":
    print(emit(os.path.join(os.path.dirname(__file__), "generated")))
