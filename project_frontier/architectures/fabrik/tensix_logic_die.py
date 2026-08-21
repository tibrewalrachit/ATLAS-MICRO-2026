"""Tensix-class logic die for Fabrik, as an ATLAS chip config (cycle-level).

Emits ATLAS chip YAMLs where each vault-core's compute tile is a Tensix
analog: 1.35 GHz, mac_num per the tile-utilization bracket, 1.5 MB SRAM.
Two bracket variants (see analytical/compute.py for rationale):
  - tinytile: mac_num=2048 (full FP8 MAC throughput; tiny-tile best case)
  - tile32_m1: mac_num=64 (2048/32; worst case for M=1 without row packing)
Power/area fields are carried from the stock test chip: this experiment
compares LATENCY/UTILIZATION of the logic-die swap at iso-memory-system,
not energy (noted in the report).
"""
import os

TEMPLATE = """architecture:
  frequency: 1350
  core_num: 16
  core:
    controller:
      power: 1.0
      area: 1.0
    matrix:
      mac_num: {mac_num}
      power: 1.1
      area: 1.1
    vector:
      vec_num: 512
      power: 1.2
      area: 1.2
    buffer:
      buffer_size: 1536
      read_bw: 8192
      write_bw: 8192
      power: 1.3
      area: 1.3
  dram:
    config_path: configs/architecture/dram/hb_dram_16ch.yaml
    power: 1.4
    area: 1.4
  noc:
    topology: mesh
    config_path: configs/architecture/noc/4x4_mesh
    flit_size: 64
    power: 1.5
    area: 1.5
"""

VARIANTS = {"tensix_tinytile": 2048, "tensix_tile32_m1": 64}


def emit(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, macs in VARIANTS.items():
        p = os.path.join(out_dir, f"chip_{name}.yaml")
        open(p, "w").write(TEMPLATE.format(mac_num=macs))
        paths[name] = p
    return paths


if __name__ == "__main__":
    print(emit(os.path.join(os.path.dirname(__file__), "generated")))
