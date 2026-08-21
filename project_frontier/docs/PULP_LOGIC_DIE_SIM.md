# PULP Logic-Die Simulator — architecture and fidelity map

Goal: a cycle-accurate simulator of the Fabrik logic die composed from
PULP-platform IPs (Snitch cluster + RedMulE + NEUREKA + FlooNoC) on CUBE
3D-DRAM stacks. Design decision (user-confirmed): **hybrid fidelity** — each
layer simulated by the tool whose accuracy claim actually holds for it,
with no simulator forks.

## Layers and fidelity labels

| layer | what runs | tool | fidelity |
|---|---|---|---|
| Snitch orchestration (SSR setup, FREP, HWPE config/trigger, DMA) | real RISC-V kernels | **Banshee** (binary translation, SSR/FREP latency model; Modal build) | cycle-approximate, executed |
| RedMulE busy time | closed form from array geometry (12 rows x 4 CEs x 4 slots, K-chunk 16) | `pulp_sim/redmule_model.py` | paper-anchored: reproduces 99.4% @96^3 exactly |
| NEUREKA busy time | closed form from BinConv dataflow (TP 32x32, bit-serial Qw, spatial PEs) | `pulp_sim/neureka_model.py` | architecture-derived; NE16-family dataflow |
| TCDM/HCI contention | 32-bank port model, pessimistic arbitration | `pulp_sim/tcdm.py` | pessimistic bound (guardrail) |
| DRAM channels (CUBE) | Ramulator HBDRAM 4ch x 64 GB/s per stack | ATLAS | cycle-accurate |
| Die NoC | BookSim mesh, FlooNoC 512-bit -> 64 B flits | ATLAS | cycle-accurate |
| Full-model decode | per-op max(tile busy, memory) composition via kernel LUT | `pulp_sim/lut.py` + `analytical/kernel_cache.py` | composed; method=pulp_tile_sim |

Orchestration constants come from Banshee measurements
(`results/processed/banshee_orchestration.json`); until the measurement run
lands they are ASSUMED with ranges and the outputs say so
(`tile_sim.orchestration()['source']`).

## Findings the cycle models produced immediately

1. **RedMulE decode GEMV requires the operand-swap mapping** (stream the
   large dimension through the 12-row axis): 99.7% array utilization
   swapped vs 8.3% naive. The SSR streamers make the swap free in traffic.
2. **NEUREKA's 36 PEs parallelize SPATIAL points, not output channels** — a
   B=1 decode GEMV activates one PE: 256 MAC/cycle at Qw=4, not the 9,216
   a peak-derived model assumes (a 9x error the quick analytical org made
   and this model corrected). Batch raises it linearly to 36 PEs.
3. **Low-bit verdict for V4-Flash experts**: NEUREKA int4 (as FP4 proxy,
   ASSUMPTION P8) runs a 4096x2048 expert slice in 32.8k cycles vs RedMulE
   fp16 175k cycles — **5.3x faster and 4x less DRAM traffic**. Four
   NEUREKAs per stack (1.02 GMAC/clk) cover a 256 GB/s stack's fp4 stream
   demand (0.512 GMAC/clk) with 2x headroom at B=1.
4. **The V4-Flash indexer scan starves the PULP die at long context**: at
   128K the composed decode is indexer-dominated (S: 155 tok/s/user) —
   64-192 tiles of RedMulE supply only 3-9 TFLOP/s fp16 against a scan that
   wants ~0.1+ PFLOP/s. Same conclusion as the Tensix study reached via a
   different route: the index scan needs a dedicated engine on any tile.

## Runbook

```bash
python project_frontier/tests/test_pulp_sim.py            # model anchors
python - <<'PY'                                            # composed decode
import sys; sys.path.insert(0, 'project_frontier/pulp_sim')
from lut import compose_v4flash_decode
print(compose_v4flash_decode(stacks=16, batch=1))
PY
# Banshee (Modal): scratchpad/banshee_modal.py -> orchestration json
```
