# Project Frontier: Feasibility of Fabrik, a 3D-DRAM Decode Accelerator

*A full-stack simulation study on the ATLAS (MICRO '26) infrastructure.*

**Evidence tags used throughout:** `[measured]` = run in this project on real
tools (Ramulator/BookSim/HotSpot cycle or thermal simulation, or public API
measurements by named third parties); `[cycle]` = ATLAS cycle-level simulation;
`[analytical]` = this project's validated analytical model; `[external]` =
sourced public figure; `[estimated]` = derived from external figures with
stated method; `[assumed]` = modeling assumption, carried through the
uncertainty analysis (§20).

---

## Abstract

We ask whether a purpose-built decode accelerator ("Fabrik") — small compute
die stacked on high-bandwidth 3D DRAM — can serve modern long-context MoE
models at Cerebras-class interactivity (1,000–2,000 output tokens/s/user) at
batch 1–4, with NVIDIA-class or better cost per token and materially better
energy per token. We build an independent analytical simulator validated to
within ±20% against ATLAS cycle-level simulation `[cycle]`, calibrate its
memory model with Ramulator measurements per access pattern `[measured]`, and
sweep 64,800 design points across bandwidth (12–150 TB/s), compute (0.25–4
PFLOP/s), capacity (128–512 GB), and SRAM (8–256 MB) for DeepSeek V4-Flash
(284B/13B-active, hybrid compressed-sparse attention) and MiniMax M3
(428B/23B-active, MiniMax Sparse Attention), both at 1M context.

**Finding: SUPPORTED for DeepSeek-V4-Flash-class workloads; PLAUSIBLE BUT
UNPROVEN for MiniMax-M3-class workloads.** A 50 TB/s / 1 PFLOP/s / 256 GB
Fabrik serves V4-Flash at 2,560 TPS/user (B=1, 128K ctx) in ~480 W,
~5.3 tok/J, and $0.16/1M output tokens at $25K system cost — a
(interactivity, cost) point roughly 8–20× faster than measured GPU serving of
the same model at comparable or lower cost/token `[analytical]`, `[external]`.
M3 reaches 2,000 TPS/user only at ~100 TB/s and ~1 kW `[analytical]`, brushing
our falsification thresholds; its 428 GB fp8 weight footprint also makes
single-device capacity the binding constraint.

---

## 1. Motivation

Autoregressive decode of MoE LLMs at batch 1–4 is memory-bandwidth-bound:
every generated token must stream the activated weights and the attention
working set. GPUs amortize weight traffic across large batches, sacrificing
per-user speed; wafer-scale SRAM machines buy speed with silicon area. A
logic-on-3D-DRAM part occupies a third position: DRAM-class capacity and cost
per bit with SRAM-class bandwidth. The commercial hypothesis is that decode —
disaggregated from GPU prefill — is the workload where that position wins.

## 2. Research question

> Can a purpose-built 3D-DRAM decode accelerator achieve ~1,000–2,000 output
> tokens/s/user at NVIDIA-class or better cost/token and materially better
> energy efficiency for modern large MoE models at batch sizes 1–4?

## 3. Workloads

Both workloads are reconstructed **exactly** from official sources (archived
with revisions in `models/source_of_truth/`; per-parameter provenance in
`models/model_manifest.json`).

### 3.1 DeepSeek V4-Flash `[external: official config + inference/model.py, rev 60d8d70]`

43 layers, hidden 4096, all layers MoE (256 routed FP4 experts, top-6, 1
shared FP8, inter 2048; first 3 layers hash-routed). Attention is
single-shared-KV MQA with head_dim 512 (rope 64), in three per-layer modes set
by `compress_ratios`: 2 sliding-window(128) layers; 21 **CSA** layers (4:1
learned-compressed KV + a 64-head×128-dim FP4 **Lightning-Indexer** that scans
*all* t/4 compressed entries and selects top-512); 20 **HCA** layers (128:1
compression, attends *all* t/128 entries). Hyper-connections (hc_mult 4)
widen the residual stream 4×. Our reconstruction totals 284.3B params
(official: 284B) and ~13.6B active (official: 13B) `[analytical, validated]`.

Decode traffic at B=1 `[analytical, validated against closed forms]`:
**11.2 GB/token** at 32K, of which routed experts are only 3.4 GB — the
*dense* attention-side projections (wq_b/wo_a/wo_b at 64×512 head geometry,
4.6B params) are the largest stream. KV is nearly flat with context
(compressed); the **indexer scan grows linearly** and reaches 0.37 GB/token at
1M — V4-Flash's context-scaling bottleneck is index-scan bandwidth, not KV.
Total KV+index state: **3.6 GB/user at 1M** — trivially small.

### 3.2 MiniMax M3 `[external: official config rev f0e1c1e; MSA mechanics from arXiv:2606.13392]`

60 layers, hidden 6144, GQA 64Q/4KV heads ×128. Layers 0–2: dense FFN
(12288) + **full attention**; layers 3–59: MoE (128 experts, top-4, 1 shared,
inter 3072) + **MSA** (top-16 blocks of 128 tokens + local block, 4 index
heads ×128, max-pool block scoring, KV-outer-gather-Q). Reconstruction: 426B
(official 428B; delta = the 7 MTP modules we model off). The checkpoint is
bf16; serving precision is unpublished — we model `exact_fp8` (base,
`[assumed]`) and `exact_bf16` variants.

Decode traffic at B=1, fp8: **26 GB/token** at 32K (routed+shared experts
12.9 GB, attention/dense projections 13 GB), rising to 29.4 GB at 1M — of
which **3.1 GB/token is the three full-attention layers alone**. Per-user KV
state: **64.5 GB (fp8) / 129 GB (bf16) at 1M** — capacity, not bandwidth, is
M3's binding constraint on a single device. In bf16, M3 does not fit any
swept capacity (needs 856 GB weights) — every bf16 point is infeasible
`[analytical]`.

## 4. Decode characterization

Bottleneck regimes (B=1) `[analytical]`:

| workload | 32K | 128K | 1M |
|---|---|---|---|
| V4-Flash | weight-bound (dense projections > experts) | weight-bound | weight-bound, index-scan emerging (24% of added traffic; compute of attention+indexer rises 4.4×) |
| M3 (fp8) | weight-bound (experts+dense) | weight-bound | weight-bound with KV-bound tail (full-attn layers = 3.1 GB/token) |

Arithmetic intensity stays at 2.2–12 FLOP/byte — an order below GPU-optimal —
confirming decode at B≤4 belongs on a bandwidth-first machine.

## 5. Analytical methodology

Per-operator `T = max(bytes/(BW·η(pattern)), FLOPs/(peak·util(org,B))) +
fixed`, dependency-serialized with double-buffered overlap inside each
operator; a global `max(ΣT_mem, ΣT_comp)` lower bound brackets pipelining
uncertainty. Batch composition: shared weight streams amortize; per-user
KV/index/attention terms scale with B; routed-expert streams follow the
expert-reuse model. Millisecond evaluation → 64,800-point sweeps in ~7 s.

**Small-batch MoE reuse (Part VII):** with E experts and top-k routing,
E[unique] = Σ_e 1−(1−p_e)^B. At B=4: V4-Flash 23.2 unique of 24 naive (3%
saving), M3 15.3/16 (4%) `[analytical]`; even Zipf-α=1 hot-expert routing
saves only ~21% `[analytical, synthetic traces — real routing traces are not
public, labeled]`. **Expert weights effectively do not amortize at B≤4** —
the premise of a bandwidth-first design holds.

## 6. ATLAS methodology

ATLAS (MICRO '26) provides the cycle-level substrate: per-core
compute/SRAM/NoC models + Ramulator2 with a custom HBDRAM 3D-DRAM device +
BookSim + HotSpot thermal (`docs/ATLAS_ARCHITECTURE.md`). We verified
(simulator/src/core/core.cpp:156) that **each core owns a private DRAM vault**
— the stock cloud chip is 16 cores × (16ch × 1024 pin @ 500 Mbps = 1.024
TB/s) = 16.4 TB/s, 128 GB. Heavy runs execute on 32-core cloud workers
(Modal) from an image of this patched tree; upstream kick-the-tires and one
DRAM-DSE family reproduce cleanly (BASELINE.md) `[cycle]`.

**Validation** (`fig13`, `results/processed/validation_summary.csv`): the
analytical model, with zero tuned parameters, reproduces ATLAS cycle-level
end-to-end decode latency within **0.82× (opt-66B dense, TP8, B=8)** and
**1.16× (Mixtral-8×22B MoE, TP8/EP8, B=8)** on the stock chip `[cycle]`.

**Memory-pattern calibration `[measured]`** (`calibrate_mem_eff.py`, Ramulator
`test_dram` on the 1 TB/s vault): well-tiled sequential weight streams achieve
**0.80** of peak; MSA-style 128-token block gathers **0.77**; V4-style
scattered 576 B row gathers **0.29**; index/HCA scans **0.81**. Naive tiling
collapses streams to 0.20–0.31 — the SRAM-sensitivity model is anchored to
this measured spread. Burst dependency latencies: 512-row CSA gather 7.7 µs,
MSA 17-block gather 2.7 µs.

## 7. Fabrik architecture

A Fabrik point is (BW, compute, capacity, SRAM, compute-org), physically
realized as N vault-cores (per-core 1.024 TB/s ATLAS-validated vault; N =
BW/1.024), per-core MAC count set by the compute target, per-core stack
density by capacity (feasible 1–64 GB/core). The flagship configurations:

| | V4-Flash flagship | M3 flagship |
|---|---|---|
| DRAM BW | 50 TB/s (49 vaults) | 100 TB/s (98 vaults) |
| Compute (FP8) | 1 PFLOP/s | 2 PFLOP/s |
| Capacity | 256 GB | 512 GB |
| SRAM | 64 MB | 128 MB |
| org | hybrid (reconfig systolic + GEMV lanes + index engine) | hybrid |

## 8. 3D-DRAM architecture

Not all bytes are equal (§6 calibration). The DSE therefore prices each
operator's traffic at its measured pattern efficiency. The **Raptor
calibration case** (`calibration/raptor/`) checks realism at scale: 104 vaults
→ 106.5 TB/s peak reproduces a documented logic-on-DRAM card class
(~105 TB/s); sustained streaming 86 TB/s (81%); ns-scale per-flit streaming
latency, consistent with the ~2.5 ns reference figure `[measured]`. All
parameter divergences from the real chip are enumerated in the module header
— this is a behavioral sanity check, not a product reproduction.

## 9. Small-batch compute architecture

`[analytical; org utilization parameters assumed, anchored to ATLAS area/power
scalars and swept ±30% in MC]` At B=1–4 the workload is GEMV-shaped. Fixed
128×128 systolic arrays waste rows (util ~12.5% at M=1); reconfigurable
aspect ratios recover to ~60–80%; dedicated GEMV lanes ~90%. Speedup of
hybrid vs the ATLAS baseline engine at B=1, 1M ctx: **1.57× for V4-Flash**
(the index engine is the difference — see §10) and **1.05× for M3**. Compute
utilization at the flagship points stays below 15% of peak FP8 — decode
compute exists to keep up with the stream, not to saturate.

## 10. Sparse-attention acceleration

V4-Flash: the indexer scan (all t/4 entries × 64 heads × 128 dims, FP4) is
86 GFLOP + 0.37 GB per token at 1M and runs poorly on generic engines
(index_util 0.3): a dedicated scan/top-k engine (util 0.9) is the main term
in the 1.57× hybrid speedup — **a specialized indexing unit is worthwhile for
V4-class CSA**. M3: MSA's block-max scoring is small (n_blocks×4×128) and its
gathers are 128 KB-contiguous (0.77 efficiency `[measured]`) — **specialized
hardware buys little (~5%)**; commodity engines suffice.

## 11. Power and thermal

Power model (per-subsystem pJ/bit + org W/TFLOP; constants `[assumed]`,
MC-swept): at the V4 flagship point, DRAM array reads dominate (≈55% of
energy), then interface, compute, SRAM, NoC — "memory-array energy cannot be
omitted" is quantitatively confirmed. HotSpot thermal `[measured]` (PyTA,
liquid cooling, 8 memory layers, 85 °C threshold): the 40–50 TB/s flagship
class at its natural ~480–560 W runs 84–92 °C — thermally valid at 1 GHz or
with one 100 MHz derate step at a 400 W envelope. Full envelope×design table:
`results/processed/thermal_summary.csv`.

## 12. Prefill/decode disaggregation

`T_request = T_prefill(GPU) + T_handoff + N_out × T_decode` with GPU prefill
55k tok/s `[estimated, labeled]`. V4-Flash handoff is trivial: 3.6 GB KV at
1M moves in 14 ms at 256 GB/s (<0.1% of a request). M3 is the stress case:
64 GB KV at 1M takes 1.0 s on a 64 GB/s link (5% of a short-output request,
TTFT +1 s) — M3 disaggregation wants ≥256 GB/s links or KV-resident session
affinity. Link latency (100 ns–10 µs) is irrelevant at these sizes
`[analytical]`.

## 13. Experimental setup

64,800-point analytical DSE (10 BW × 8 compute × 5 capacity × 6 SRAM × 2–3
model modes × 3 contexts × 3 batches), calibrated as §6; representative-point
cycle/ thermal validation; serving simulation (continuous batching, cap 4);
TCO per Part XXIII formulas. All reproducible: `experiments/run_dse.py`,
`run_pareto.py`, `run_min_hw.py`, `run_thermal.py`, `run_serving.py`,
`run_orgs.py`, `run_disagg.py`, `run_uncertainty.py`, `make_figures.py`.

## 14. Performance results

Flagship points `[analytical]`:

| config | ctx | B | decode step | TPS/user | aggregate |
|---|---|---|---|---|---|
| V4 flagship (50 TB/s) | 128K | 1 | 0.39 ms | **2,562** | 2,562 |
| V4 flagship | 128K | 4 | 0.67 ms | **1,484** | 5,937 |
| V4 flagship | 1M | 1 | 0.46 ms | **2,184** | 2,184 |
| M3 flagship (100 TB/s) | 128K | 1 | 0.43 ms | **2,310** | 2,310 |
| M3 flagship | 128K | 4 | 0.92 ms | **1,088** | 4,352 |
| M3 flagship | 1M | 1 | 0.49 ms | **2,040** | 2,040 |

Minimum hardware for targets (B=1/B=4 @128K) `[analytical]`: V4-Flash
1,000 TPS: **24/32 TB/s**; 2,000 TPS: **40/75 TB/s**. M3-fp8 1,000 TPS:
**40/100 TB/s**; 2,000 TPS: **100 TB/s / not reachable at B=4 within the
swept space**. Serving simulation: at ~50–70% device occupancy the V4
flagship holds p50 ≈ 2,200 TPS/user with p95 TPOT < 0.7 ms; saturation
degrades to the B=4 floor (~1,480) with queueing, not TPOT, absorbing load.

## 15. Energy results

V4 flagship: 5.3 tok/J at B=1, 10.7 tok/J at B=4 (128K); M3 flagship 2.3 /
3.7 tok/J. GPU reference: an H200/B200-class node serving V4-Flash at
measured provider speeds implies ~0.5–1 tok/J `[estimated from external
throughput and node power, labeled]` — Fabrik's advantage is roughly **5–10×
at iso-model** at B=4. Subsystem split at flagship: DRAM array 55%,
interface 9%, compute 7%, SRAM 6%, static 9%, rest NoC/controller.
Reducing bandwidth below the interactivity-minimum always improves tok/J
slightly but costs TPS/user; the energy-optimal point sits just above the
target-TPS bandwidth minimum (§17, fig04).

## 16. Cost / TCO results

At $25K system cost, 4-year life, PUE 1.25, $0.08/kWh, 60% utilization
`[assumed, swept]`: V4 flagship **$0.165/1M output tokens at B=1** and
**$0.072/1M at B=4**; M3 flagship $0.19/$0.10. Measured GPU serving of the
same model spans $0.168–0.66/1M `[external]`. NRE: even $400M amortizes below
GPU-reference cost/token by ~10k units at B=4 economics; $50M by ~1k units
`[analytical]`.

## 17. Design-space exploration

fig02: at B=1 the TPS contours are nearly vertical — **bandwidth is the only
first-order axis**; ≥0.5 PFLOP/s of compute suffices until 1M-context
attention/indexer FLOPs matter. At B=4 contours bend: 2,000 TPS needs both
(75 TB/s and ~1 PFLOP/s for V4). Extra bandwidth beyond the target minimum
buys headroom linearly until compute-bound onset (V4 B=4: ~125 TB/s at 2
PFLOP/s). 100 TB/s is *not* overprovisioned for M3 (it is the 2,000-TPS
minimum) but is ~2.5× overprovisioned for V4-Flash at B=1.

## 18. Pareto frontier

3,481 non-dominated points (`results/pareto/`). The 2,000-TPS frontier for
V4 is anchored by 40–75 TB/s / 0.25–1 PFLOP/s / 32–64 MB designs at 380–750 W
— i.e., the flagship class, not the 150 TB/s corner. SRAM beyond 64 MB buys
<2% at 50 TB/s (fig10); capacity is pure feasibility (V4: 128 GB suffices to
B=4 @1M; M3: 512 GB admits only B=1 @1M).

## 19. Comparison with existing accelerators

`baselines/external.csv` (all sourced; comparison classes labeled). Same-model
GPU serving: 107–306 TPS/user, $0.168–0.66/1M. Cerebras: 1,800–4,400 TPS/user
on *different* (dense/smaller) models `[external, market reference]` at
$0.85–1.20/1M-class prices; no public Cerebras/Groq offering of V4-Flash/M3
exists, so same-model comparison with wafer-scale is not possible — the
conclusion rests on the same-model GPU comparison plus Fabrik's absolute
numbers, and survives this gap.

## 20. Sensitivity analysis

2,000-sample Monte Carlo on the V4 flagship (10 variables, triangular
low/base/high): TPS/user p5–p95 = **2,103–2,689** (target 2,000 holds at p5);
tok/J 3.4–5.8; $/1M 0.14–0.35; power 400–739 W. Top-5 drivers (by $/1M
swing): hardware cost, fleet utilization, sequential-stream efficiency, DRAM
array pJ/bit, compute utilization scale. No sampled combination pushed the
V4 flagship below 1,900 TPS/user.

## 21. Limitations

1. V4/M3 cycle-level validation is by proxy: the analytical model is
   cycle-validated on ATLAS-native models (±20%) and pattern-calibrated on
   Ramulator; the exact V4/M3 operator DAGs run analytically only.
2. Compute-org utilizations and all power constants are assumptions (swept);
   no RTL exists.
3. ATLAS itself ships anonymized parameters (its AE docs); absolute cycle
   numbers are trend-faithful, not silicon casting.
4. Routing traces are synthetic; real V4/M3 expert distributions unpublished.
5. M3 serving precision (fp8) is our assumption; bf16 makes M3 infeasible on
   a single device of any swept capacity.
6. GPU tok/J and prefill throughput references are estimates from public data.
7. Multi-device Fabrik (which would relieve M3 capacity) is future work.

## 22. Conclusions — answers to the mandated questions

**Architecture.** (1) Min BW for 1,000 TPS/user: 24 TB/s (V4, B=1) / 40 TB/s
(M3-fp8, B=1). (2) For 2,000: 40 / 100 TB/s. (3) ~100 TB/s is required for
M3-2000 but ~2.5× overprovisioned for V4 B=1. (4) Bandwidth stops helping at
the compute-bound knee (V4 B=4: ~125 TB/s @2 PFLOP/s) — and earlier, at the
target, for energy. (5) 0.25–0.5 PFLOP/s suffices at B=1; ~1–2 PFLOP/s at
B=4/1M. (6) Compute-bound only at B=4 + long context + low compute. (7)
32–64 MB SRAM; beyond that <2%. (8) Yes for B=1–4: hybrid GEMV/reconfig
gives 1.45–1.57× over a fixed engine (V4). (9) V4: yes (index engine is most
of the win); M3: no (~5%).

**Workload.** (10) V4-Flash: weight-bound at 32K/128K; index-scan becomes
the growth term at 1M (KV stays negligible). (11) M3: weight-bound at
32K/128K; at 1M the 3 full-attention layers make it KV-heavy (3.1 GB/token)
and capacity-bound (64 GB/user). (12) **V4-Flash is the better Fabrik
match** — 2.3× less traffic/token, 18× less KV state, fits 256 GB.

**Interactivity.** (13–16) V4: 1,000 TPS at B=1 ✓ (24 TB/s), B=2 ✓, B=4 ✓
(32 TB/s); 2,000 ✓ (40–75 TB/s). M3-fp8: 1,000 ✓ (40–100 TB/s); 2,000 only
at B=1 with 100 TB/s. (17) B=1→4 degradation: V4 −42% (2,562→1,484); M3
−53% (2,310→1,088).

**Power.** (18) V4: yes — 2,000 TPS/user at 377–750 W. M3: 2,000 TPS needs
~900–1,000 W: *outside* 500–800 W. (19) 5.3–10.7 tok/J (V4), 2.0–3.7 (M3).
(20) DRAM array (~55%). (21) Yes: past the target-TPS minimum, more
bandwidth costs tok/J for nothing the SLA needs.

**Economics.** (22) ≤$40K matches the cheapest same-model GPU route at B=1;
at B=4, even $100K does. (23) $0.072–0.165/1M (V4 flagship, 60% util). (24)
Batching 1→4 cuts $/token 2.3× while holding 1,484 TPS/user. (25) $50M NRE
amortizes at ~1k units; $400M by ~10k units.

**(26) Thesis verdict:**

> *A memory-centric accelerator can provide Cerebras-class interactivity with
> NVIDIA-class or better economics and materially better energy efficiency
> for B=1–4 decode.*

**SUPPORTED** for DeepSeek-V4-Flash-class workloads (compressed-attention,
low-active-parameter MoE): every element of the claim holds simultaneously at
the 50 TB/s flagship point, with p5 Monte-Carlo margin above the 2,000-TPS
target and cost/token at-or-below the best same-model GPU route.
**PLAUSIBLE BUT UNPROVEN** for MiniMax-M3-class workloads: interactivity and
economics hold at B=1 with fp8 weights, but the 2,000-TPS point needs ~1 kW
(falsification criterion grazed), B=4 @1M violates single-device capacity,
and the fp8-serving assumption is unverified. The falsification checklist
(Part XXXIII) fires on no V4 criterion and on two M3 criteria (power;
capacity-dominated economics at 1M).

### Falsification checklist (Part XXXIII)

| criterion | V4-Flash | M3-fp8 |
|---|---|---|
| <25% usable BW at B=1–4 | no (51–63% modeled, patterns measured 29–81%) | no (60–70%) |
| >4–5 PFLOP/s needed to consume BW | no (≤1–2) | no (≤2) |
| >1 kW for target TPS | no (≤750 W) | **yes at 2,000 TPS (~1 kW)** |
| large B=4 TPS/user degradation | −42% (acceptable, still >1,000) | −53% (1,088 at 128K; fails @1M capacity) |
| capacity/package dominates TCO | no | **partially (512 GB, B=1-only @1M)** |
| communication dominates sub-ms decode | no (<0.1%) | no at ≥256 GB/s links |
| specialized compute buys little | no (1.57×) | yes (1.05×) — commodity engines fine for M3 |
| $/token worse than GPU serving | no (at/below best route) | no (B=1 parity, B=4 better) |

---

## Appendix A — DeepSeek-V3 characterization and card balance (FabrikSim incorporation)

An independent analysis line ("FabrikSim", separate session, DeepSeek-V3 671B/37B
as characterization vehicle) was incorporated into this codebase
(`models/deepseek_v3/`, `analytical/balance.py`, `economics/tiers.py`,
`experiments/run_balance.py`). Its headline numbers were re-derived here from
the official V3 config with an independently-written DAG and **reproduce
exactly** `[analytical, cross-validated]`: 671.0B parameters; 37.0 GB/step at
B=1/8K (FabrikSim: 36.9); MLA absorbed-form arithmetic intensity **484
FLOP/byte**; attention-FLOP crossover at ~4,400 tokens of context (FabrikSim:
~4,300); attention = 97% of decode FLOPs at 128K. Automated in
`tests/test_workloads.py::test_v3_totals_and_mla`.

### A.1 The MLA finding and its consequence for Fabrik

Decode splits cleanly: **bandwidth is experts/FFN (2–8 FLOP/byte); FLOPs are
attention (484 FLOP/byte under MLA)**. Machine balance (dense-FP8/BW): B300 ≈
562 — barely memory-bound on MLA; H200 ≈ 417 — already compute-bound; Fabrik
at 2 PFLOP/s / 105 TB/s ≈ 19 — compute-bound on MLA by 25×. At a 2 PFLOP/s
compute budget, 484-FLOP/byte attention can consume only **4.1 TB/s** of KV
bandwidth. Consequences, now verified in this framework:

1. **Do not build KV bandwidth into the card.** Provision bandwidth per
   operator by arithmetic intensity, not byte share: expert streams get the
   3D-DRAM system (and need only ~0.2–0.8 PFLOP/s to consume 105 TB/s);
   MLA-class attention gets a large tile array fed from single-digit TB/s.
   This is the strongest argument for heterogeneous tiles, and it is why the
   Fabrik flagship pairs a bandwidth-first vault array with a modest but
   attention-provisioned compute budget.
2. **Dense-MLA long context misses the interactivity target on any
   bandwidth-first card**: V3 on a 105 TB/s / 2 PFLOP/s Fabrik reaches only
   ~1,600 TPS/user at 8K but **~520 at 128K (B=1)** — compute-bound, not
   bandwidth-bound `[analytical]`. The V4-Flash generation *deleted this
   problem in the algorithm*: CSA/HCA cap both KV reads and attention FLOPs,
   which is precisely why the §14 flagship results hold to 1M context. The
   thesis is therefore **generation-dependent**: SUPPORTED for
   compressed-attention MoE (V4-Flash-class), NOT ACHIEVED for dense-MLA
   V3-class beyond ~32K context on this compute budget — and the exposure
   flagged in §21 (a market swing back to GQA/dense-MLA long-context models)
   is quantified by exactly this gap. Sensitivity knob:
   `deepseek_v3.build_decode_dag(ctx, attn_read_fraction=f)` models
   sparse-attention retrofits; the DSE can sweep it.

### A.2 Card balance: capacity per bandwidth is the ratio to sweep

Workload-demanded GB of capacity per TB/s of bandwidth (V3@8K, fp8)
`[analytical, matches FabrikSim to 0.3%]`:

| | T=1000 | T=2000 | T=4000 |
|---|---|---|---|
| B=1 | 18.1 | 9.1 | 4.5 |
| B=4 | 7.0 | 3.5 | 1.8 |
| B=16 | 2.4 | **1.2** | 0.6 |
| B=64 | 1.1 | 0.6 | 0.3 |

A 128 GB / 105 TB/s card (ratio 1.22) is balanced for **B=16 @ 2,000 TPS**,
not B=1–4: at B=1 it strands ~90% of its bandwidth on capacity grounds. The
V4-Flash flagship chosen in §7 (256 GB / 50 TB/s = 5.1 GB per TB/s) sits
between the B=1 and B=4 demand rows for its own workload — consistent with
the min-hardware table in §14. Balance ratio is now a first-class output of
the DSE (`analytical/balance.py:demanded_balance`).

### A.3 Why wafer-scale SRAM cannot price frontier MoE

Routing is data-dependent, so a resident architecture holds **all** weights:
at ~$57–65k/GB of SRAM capacity, a 671 GB model implies a **$56M capex floor
per replica** (16 CS-3 wafers at wafer granularity) before the first token
`[estimated from public figures]` — consistent with the observed market
structure (public SRAM rate cards top out ~120B; frontier MoE is
custom-priced dedicated endpoints only). Memory-tier summary
(`economics/tiers.py`, all labeled estimates): SRAM ~$61k/GB & ~$155/(TB/s);
HBM3e ~$156/GB & ~$5,600/(TB/s); hybrid-bonded 3D DRAM ~$200–600/GB &
~$300–1,000/(TB/s) — the only tier cheap on *both* axes, which is what a
20×-sparse MoE at low batch requires. NAND in-flash compute holds capacity
($0.10/GB) but is throughput-bound ~280× short for this regime; its real
role is capacity tiering (98% of a model behind a ~12 GB DRAM working set),
not decode bandwidth.

### A.4 GPU contention: the bound this study can produce without a GPU simulator

The GPU comparison in §16/§19 uses *measured* provider throughput. The
ideal-scaling roofline for an 8×B300-class group on V4-Flash (11.2 GB/token
over 8×8 TB/s at 0.8 efficiency) is ≈ **4,570 TPS/user**; the best measured
provider serves **306** `[external]`. The ≥**15×** gap between roofline and
measurement is the contention term (collectives, expert dispatch, launch
overhead) that GPU serving pays and Fabrik's single-device design does not.
We report it as a bound from public data: a cycle-accurate GPU contention
number requires a GPU microarchitecture simulator, which is outside ATLAS's
scope — flagged as external future work, not silently estimated.
