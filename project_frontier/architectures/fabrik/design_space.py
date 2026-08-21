"""Fabrik parameterized design space (Part X) and physical-parameter mapping.

Abstract sweep axes -> physical realization:
- Bandwidth: HBDRAM channels of 1024 pins @ 2 Gbps/pin = 256 GB/s/channel
  (ATLAS's HBDRAM_500Mbps preset scaled 4x; the timing preset for Fabrik runs
  is generated per-config; the Raptor calibration case validates the scaling).
  channels = ceil(BW / 256 GB/s).
- Capacity: per-channel stack density; capacity_GB / channels must fall in
  the per-channel density range of the HBDRAM org presets (0.125-4 Gb x banks);
  configs where it cannot are flagged infeasible_capacity.
- Compute: cores x mac_num x frequency. Fabrik base frequency 1 GHz (thermal
  model may derate); mac_num per core fixed at 16384 FP8 MAC/cycle
  => per-core 32.8 TFLOP/s; cores = ceil(PFLOPs / 0.0328).
- SRAM: split evenly across cores (ATLAS buffer_size is per-core KB).
"""
import math, itertools, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analytical"))
from model import FabrikPoint

BW_GRID_TBPS = [12, 24, 32, 40, 50, 60, 75, 100, 125, 150]
COMPUTE_GRID_PFLOPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
CAPACITY_GRID_GB = [128, 192, 256, 384, 512]
SRAM_GRID_MB = [8, 16, 32, 64, 128, 256]

CH_BW_GBPS = 256.0          # 1024 pins x 2 Gbps
CORE_TFLOPS = 32.768        # 16384 MAC x 2 x 1 GHz


def physicalize(bw_tbps, pflops, cap_gb, sram_mb, org="hybrid"):
    channels = math.ceil(bw_tbps * 1000 / CH_BW_GBPS)
    cores = max(1, math.ceil(pflops * 1000 / CORE_TFLOPS))
    gb_per_ch = cap_gb / channels
    # per-channel density realizable range (stacked banks, 1-32 Gb x 8 stack)
    feasible_cap = 0.5 <= gb_per_ch <= 16.0
    return {
        "channels": channels, "pins_per_channel": 1024, "gbps_per_pin": 2.0,
        "cores": cores, "mac_per_core": 16384, "frequency_MHz": 1000,
        "sram_kb_per_core": sram_mb * 1024 // cores,
        "gb_per_channel": round(gb_per_ch, 2), "feasible_capacity": feasible_cap,
    }


def make_point(bw, pf, cap, sram, org="hybrid") -> FabrikPoint:
    return FabrikPoint(f"fabrik_bw{bw}_c{pf}_m{cap}_s{sram}_{org}",
                       bw, pf, cap, sram, org=org)


def full_grid(orgs=("hybrid",)):
    for bw, pf, cap, sram, org in itertools.product(
            BW_GRID_TBPS, COMPUTE_GRID_PFLOPS, CAPACITY_GRID_GB, SRAM_GRID_MB, orgs):
        phys = physicalize(bw, pf, cap, sram, org)
        yield make_point(bw, pf, cap, sram, org), phys
