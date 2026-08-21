# PULP Logic-Die Simulator — architecture and fidelity map

Goal: a cycle-accurate simulator of the Fabrik logic die composed from
PULP-platform IPs (Snitch cluster + RedMulE + NEUREKA + FlooNoC) on CUBE
3D-DRAM stacks. Design decision (user-confirmed): **hybrid fidelity** — each
layer simulated by the tool whose accuracy claim actually holds for it,
with no simulator forks.

## Layers and fidelity labels

| layer | what runs | tool | fidelity |
|---|---|---|---|
| Snitch orchestration (SSR setup, FREP, HWPE config/trigger, DMA) | real RISC-V kernels | **Banshee** — *not obtainable in this environment, see below* | **ASSUMED constants with ranges** |
| RedMulE busy time | closed form from array geometry (12 rows x 4 CEs x 4 slots, K-chunk 16) | `pulp_sim/redmule_model.py` | paper-anchored: reproduces 99.4% @96^3 exactly |
| NEUREKA busy time | closed form from BinConv dataflow (TP 32x32, bit-serial Qw, spatial PEs) | `pulp_sim/neureka_model.py` | architecture-derived; NE16-family dataflow |
| TCDM/HCI contention | 32-bank port model, pessimistic arbitration | `pulp_sim/tcdm.py` | pessimistic bound (guardrail) |
| DRAM channels (CUBE) | Ramulator HBDRAM 4ch x 64 GB/s per stack | ATLAS | cycle-accurate |
| Die NoC | BookSim mesh, FlooNoC 512-bit -> 64 B flits | ATLAS | cycle-accurate |
| Full-model decode | per-op max(tile busy, memory) composition via kernel LUT | `pulp_sim/lut.py` + `analytical/kernel_cache.py` | composed; method=pulp_tile_sim |

### Banshee: why the orchestration layer stays ASSUMED

Banshee could not be built or obtained here, and the reason is structural
rather than incidental — recorded so it is not re-attempted blindly:

1. `banshee/build/runtime.rs` compiles the JIT runtime with
   `-Cllvm-args=-opaque-pointers=0`. That flag is consumed by **rustc's own
   bundled LLVM**, and it was removed when opaque pointers became mandatory
   in LLVM 17. So the README's `rustc 1.67.0` pin is a **hard requirement**,
   not a tested-with note; no modern toolchain can build it.
2. `banshee/rust-toolchain.toml` pins 1.67.0 and silently overrides rustup's
   default toolchain, so toolchain selection must be done by deleting it.
3. Its `Cargo.lock` is stale against its manifest, so cargo must re-resolve —
   and **no rustc-1.67-compatible dependency tree still exists**. Cargo's
   MSRV-aware resolver (1.84+, `resolver.incompatible-rust-versions =
   fallback`) correctly downgraded `thiserror`, `termion` and others, but
   `syn v3.x` requires >= 1.71 in every published version and something in
   the tree demands `syn ^3`. This is unfixable by pinning.
4. The project's container (`ghcr.io/pulp-platform/snitch_cluster:main`)
   does **not** ship banshee. It does ship `/tools/riscv-llvm` and Verilator.
   (Its ENTRYPOINT is verilator, which swallows a `python -u` bootstrap —
   clear it with `.entrypoint([])` if running that image under Modal.)

**Consequence, stated plainly:** orchestration constants in
`pulp_sim/tile_sim.py` are ASSUMED with low/base/high ranges (ASSUMPTION
P10), and every result composed through `lut.py` carries
`orchestration_source = "ASSUMED (banshee not yet run)"`. No headline result
depends on them: they add fixed per-invocation cycles to kernels whose busy
time is dominated by the paper-anchored HWPE models.

**The viable future path** is not Banshee but the container's own contents:
`/tools/riscv-llvm` can compile Snitch orchestration kernels and Verilator
can simulate the Snitch cluster RTL directly — a genuinely cycle-accurate
route that skips the unbuildable binary-translation simulator entirely.

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
