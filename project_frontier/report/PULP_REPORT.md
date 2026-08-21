# PULP tile (Snitch + RedMulE + FlooNoC) on CUBE — Phase report

Brief: model a PULP-based Fabrik tile in ATLAS, parameters sourced from the
PULP repos, and compare directly against the Tensix-on-CUBE design points.
Repos: snitch_cluster @f78a978, redmule @7fa9fbe, FlooNoC @2fa02eb.
All parameters: `params/pulp_tile.yaml` (per-value sourcing);
assumptions with ranges: `params/ASSUMPTIONS.md`. Phase 0 baseline was
established earlier (`BASELINE.md`, unmodified upstream results).

## Tile definition (sourced)

One tile = 1 Snitch cluster (8 compute + 1 DMA cores, 128 KiB TCDM / 32
banks × 64-bit = 256 B/clk, 3 SSRs/core, FREP) + 1 RedMulE (default 12×4 =
48 CEs, FP16 FMA each; **FP8 is cast to FP16 — no FP8 rate doubling**,
sourced from the RedMulE README/pkg). FlooNoC wide channel = 512-bit →
BookSim flit 64 B. One ATLAS core = one CUBE stack (4×64 GB/s channels =
256 GB/s, reusing the stock `cloud_0.25TBps` HBDRAM preset — preset math
verified) = 4 tiles. Design points S/M/L = 64/128/192 tiles over 16/32/48
stacks, matching the Tensix study. A parametrically-legal scaled RedMulE
(16 high × 32 wide = 512 CEs, within the WIDTH ≤ HEIGHT×PIPE_REGS bound; no
silicon reference — ASSUMPTION P3) is modeled alongside the default.

## Headline: tok/s/user (analytical, calibrated efficiencies, B=1/4 @8K)

| workload | pulp_default S | pulp_scaled S | tensix S | pulp_scaled L | tensix L |
|---|---|---|---|---|---|
| Qwen3-30B-A3B B=1 | 317 | 901 | 1,154 | 2,829 | 2,946 |
| Qwen3-30B-A3B B=4 | 85 | 383 | 498 | 1,288 | 1,388 |
| V4-proxy (f=1.0) B=1 | 131 | 214 | 282 | 781 | 803 |
| V4-proxy (f=0.1) B=1 | 142 | 217 | 284 | 792 | 806 |

Achieved-vs-peak on Qwen3-30B S/B=1: pulp_default **DRAM 20% / matrix 65%**;
pulp_scaled DRAM 57% / matrix 17%; tensix DRAM 73% / matrix 6%.
(V4-Flash full model does not fit any V1 card — 213 GB > 192 GB — rows are
bandwidth-shape studies via the labeled proxy, not deployable configs.)

## Where the PULP tile loses to Tensix, and by how much

1. **Stock RedMulE starves the stack.** 4 tiles × 48 CEs = 384 GFLOP/s
   (FP16-rate) against a 256 GB/s stack needing ~512 GFLOP/s at 2 FLOP/byte:
   the default tile is **matrix-bound at 65% utilization with the DRAM at
   20%** — it wastes ~3.6× of the memory system (317 vs 1,154 tok/s, −73%).
   This is the single decisive result: the default 12×4 array is an
   edge-scale engine, not a decode-stream consumer.
2. **No FP8 advantage.** RedMulE's cast-to-FP16 architecture means FP8/FP4
   weights buy capacity but no rate, while Tensix runs FP8 natively at
   2,048 MAC/clk. At equal CE count this is a further 1× vs 1× (no gap) only
   because our Tensix model also holds FP8=peak; a Tensix FP4-double path
   would widen it.
3. **Scaled RedMulE (512 CE) closes most of the gap** — within 22% of Tensix
   at S (901 vs 1,154) and within 4% at L (2,829 vs 2,946), because at L the
   per-tile bandwidth share drops and both designs become memory-bound. But
   the scaled array has no silicon reference (ASSUMPTION P3): timing closure
   of a 16-high lock-step array and the 1,024-bit TCDM port are unproven.
4. **Where PULP wins:** control flexibility (SSR/FREP streams fit
   expert-GEMV streaming naturally — reflected in the 0.75 M=1 utilization
   vs Tensix tile granularity), open RTL end to end, and TCDM banking that
   matches the 4-channel stack interface. None of these shows up as tok/s
   at these design points.

**Bottom line:** on CUBE-class bandwidth the tile compute question is "can
the tile consume 256 GB/s?" — Tensix (4.1 TFLOP/s FP8/core) answers yes with
20× headroom; the default PULP tile (96 GFLOP/s FP16 matrix) answers no.
A PULP-based Fabrik v1 requires the scaled RedMulE variant as a *silicon
project*, whereas Tensix-class tiles exist today.

## kv_read_fraction sensitivity (V4 proxy)

At 8K context the fraction barely matters (attention ≪ weights). The
proxy's value is at 128K+: fraction 1.0→0.1 recovers ~8–12% of step time at
128K on S-class bandwidth (full sweep in `results/frontier_pulp.csv`).
Consistent with the main report: compressed attention is what keeps
long-context decode on the memory-bound side where CUBE bandwidth pays.

## Method & cross-check

Analytical model = the validated Project Frontier framework (±20% vs ATLAS
cycle sim on dense+MoE; Ramulator-calibrated pattern efficiencies; MoE
unique-experts model — B never silently amortizes expert weights; per-user
and aggregate tok/s reported separately in `results/frontier_pulp.csv`).
ATLAS cycle runs of the PULP-analog chip (both RedMulE variants, 16-stack
memory system) are queued behind the current Modal batch; cycle columns are
appended here on completion. Guardrail honored: no simulator forks — the
PULP chip is pure configuration (`architectures/fabrik/pulp/`).
