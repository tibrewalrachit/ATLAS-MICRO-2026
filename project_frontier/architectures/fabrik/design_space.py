"""Fabrik parameterized design space (Part X) and physical-parameter mapping.

Abstract sweep axes -> physical realization:
VERIFIED ATLAS STRUCTURE (simulator/src/core/core.cpp:156): every core
instantiates its own DRAMWrapper -> the chip is an array of compute tiles,
each on its own 3D-DRAM vault stack. Chip BW = cores x per-core vault BW.
The stock cloud chip (test_chip_16ch) is 16 cores x (16ch x 1024pin x
500Mbps = 1.024 TB/s) = 16.4 TB/s and 16 x 8 GB = 128 GB.

Fabrik mapping:
- Per-core vault: 16 channels x 1024 pins @ 500 Mbps = 1.024 TB/s (reuses the
  validated ATLAS HBDRAM preset unchanged; higher-rate variants only for the
  Raptor calibration case).
- cores_mem = ceil(BW / 1.024 TB/s); cores = max(cores_mem, 4).
- Compute: mac_num per core sized to meet the compute target at 1 GHz:
  mac_num = PFLOPs*1e6 / (cores * 2 * 1000MHz); rounded up to power-of-2-ish;
  flagged infeasible if mac_num > 65536 (area) or < 256.
- Capacity: per-core stack GB = capacity/cores; feasible if 1..64 GB
  (HBDRAM densities 2-32 Gb x 8-16 high stack).
- SRAM: split evenly across cores.
"""
import math, itertools, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analytical"))
from model import FabrikPoint

BW_GRID_TBPS = [12, 24, 32, 40, 50, 60, 75, 100, 125, 150]
COMPUTE_GRID_PFLOPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
CAPACITY_GRID_GB = [128, 192, 256, 384, 512]
SRAM_GRID_MB = [8, 16, 32, 64, 128, 256]

CORE_VAULT_TBPS = 1.024     # 16ch x 1024pin x 500Mbps (stock ATLAS preset)


def physicalize(bw_tbps, pflops, cap_gb, sram_mb, org="hybrid"):
    cores = max(4, math.ceil(bw_tbps / CORE_VAULT_TBPS))
    mac_per_core = pflops * 1e6 / (cores * 2 * 1000)  # MACs at 1 GHz, fp8
    mac_per_core = max(256, 1 << math.ceil(math.log2(max(mac_per_core, 1))))
    gb_per_core = cap_gb / cores
    feasible_cap = 1.0 <= gb_per_core <= 64.0
    feasible_mac = mac_per_core <= 65536
    return {
        "cores": cores, "channels": cores * 16, "pins_per_channel": 1024,
        "gbps_per_pin": 0.5, "mac_per_core": mac_per_core,
        "frequency_MHz": 1000,
        "sram_kb_per_core": sram_mb * 1024 // cores,
        "gb_per_core": round(gb_per_core, 2),
        "feasible_capacity": feasible_cap and feasible_mac,
    }


def make_point(bw, pf, cap, sram, org="hybrid") -> FabrikPoint:
    return FabrikPoint(f"fabrik_bw{bw}_c{pf}_m{cap}_s{sram}_{org}",
                       bw, pf, cap, sram, org=org)


def full_grid(orgs=("hybrid",)):
    for bw, pf, cap, sram, org in itertools.product(
            BW_GRID_TBPS, COMPUTE_GRID_PFLOPS, CAPACITY_GRID_GB, SRAM_GRID_MB, orgs):
        phys = physicalize(bw, pf, cap, sram, org)
        yield make_point(bw, pf, cap, sram, org), phys
