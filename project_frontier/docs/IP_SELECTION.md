# IP selection for realistic MoE decode simulation in ATLAS
### (Qwen3-30B-A3B and DeepSeek V4-Flash as the target workloads)

Decision document. Grounded in measured results from this project, not
vendor claims. TL;DR at the end.

## 1. What "realistic" actually depends on — the fidelity budget

Every measurement this project has produced says the same thing: **MoE
decode at B=1-4 is memory-dominated, so simulation realism is bought in the
memory system, not the compute tile.** Evidence:
- Compute utilization at CUBE-class BW (4-12 TB/s): 3-10% of peak
  (fabrikv1_cube.csv). Even at the 50 TB/s flagship: 11-31%.
- Achieved-vs-peak DRAM spread by access pattern: 0.29-0.81 (Ramulator-
  measured) - a 2.8x realism factor that dwarfs any 20-30% compute-model
  error when memory-bound.
- The one compute-bound exception: V4-Flash's indexer scan at >=128K ctx
  (every tile study - Tensix, PULP hetero - independently hit it).

So the fidelity budget should be spent, in order:
  (1) DRAM channel/bank behavior per access pattern,
  (2) expert-routing -> traffic (unique experts, placement, gather),
  (3) the index/scan path (V4-Flash only),
  (4) compute-tile utilization on skinny shapes (a scalar, not a timing
      problem, whenever memory-bound).

## 2. Scorecard of the IPs evaluated in this project

| IP / repo | maps onto ATLAS | realism for Qwen3-30B | realism for V4-Flash | provenance | verdict |
|---|---|---|---|---|---|
| **ATLAS native HBDRAM (Ramulator) + BookSim** | is ATLAS | the realism driver | same + gather patterns measured | silicon-trend-validated (paper) | **keep as the core; highest leverage** |
| **Tensix (Blackhole params)** | 1:1 (MAC pool + SRAM + NoC = ATLAS core) | high - right compute scale (4.1 TF/core), tiny-tiles cover B=1-4 | high at <=32K; misses index engine | public silicon specs; ttsim = functional only (no timing) | **primary compute tile** |
| **Snitch cluster (+Banshee)** | control layer, not the core model | orchestration constants only | same | open RTL + executable (Banshee) | **keep for orchestration realism** |
| **RedMulE** | clean (busy-time -> mac_num) but 48 CEs starve a stack (matrix-bound 65%, DRAM 20%) | too small at default; scaled variant unproven | fp16-only path; no low-bit rate | open RTL + paper (99.4% anchor reproduced) | attention engine in hetero tile only |
| **NEUREKA** | clean via closed form; spatial PEs -> B=1 uses 1 PE (256 MAC/c) | fp4 experts: adequate (2x headroom/stack) | the fp4 expert path (5.3x vs RedMulE) but integer (P8 caveat) | open RTL; NE16 lineage | expert engine in hetero tile; energy/area reference |
| **FlooNoC** | trivially (BookSim flit=64B) | fine | fine | open RTL + paper | **adopted (config-level)** |
| **ttsim / tensix-isa-simulator** | n/a (functional golden model, no timing) | - | - | official | rejected for timing, documented |

## 3. The recommendation

**Layered stack, each layer the best-provenance IP that ATLAS can express:**

1. **Memory system: ATLAS native (Ramulator HBDRAM vaults + BookSim).**
   This is where realism lives. Invest the next effort here, specifically
   for MoE: (a) expert-weight *placement* experiments via ATLAS's
   data_placement.yaml (per-channel expert affinity - can a router-selected
   expert stream from its home vault, or does it cross the NoC?); (b) the
   paged-KV random-slot path for CSA/MSA gathers (already calibrated:
   0.29/0.77); (c) the achieved-vs-peak runs now in flight.
2. **Compute tile: Tensix-parameterized ATLAS cores as the primary.**
   Not because Tensix is open - PULP wins provenance - but because it is
   the only evaluated tile at the right scale per vault (a real product's
   parameters beat a cycle-exact model of an engine 40x too small, when
   the workload is memory-bound and the tile question reduces to "can it
   drink the stream"). Tiny-tile bracket covers the B=1-4 risk.
3. **PULP IPs where they are genuinely best-in-class:**
   - **Snitch/SSR/FREP via Banshee**: executable orchestration overhead -
     the only *executed* (vs modeled) numbers in the stack; they apply to
     any tile choice.
   - **NEUREKA**: the open low-bit expert engine. For fp4 expert streams
     it is the credible open-RTL answer (with the integer/e2m1 caveat P8)
     and the anchor for energy/area claims a Tensix black box cannot give.
   - **FlooNoC**: NoC parameters for BookSim.
   - **RedMulE**: fp16 attention/dense engine in the open-RTL hetero tile;
     requires the operand-swap mapping (99.7% vs 8.3%) - now modeled.
4. **The missing IP is an index/scan engine - for either ecosystem.**
   V4-Flash's Lightning-Indexer (fp4 dot-product scan + top-k over t/4
   entries) is the one operator that is compute-bound at long context and
   has no good home on Tensix or PULP tiles (1.57x whole-die speedup when
   added, Appendix B; PULP die indexer-bound at 128K). In ATLAS it is
   expressible today as a vector-engine parameterization (vec_num sized to
   the scan rate) - and the HWPE template (hwpe-doc, the same framework as
   RedMulE/NEUREKA) is the natural path to actually building one. This is
   the highest-value *new* IP for V4-Flash realism; for Qwen3-30B (plain
   GQA) it is unnecessary.
5. **What NOT to add:** GPU-microarchitecture simulators (out of ATLAS
   scope; the GPU side stays a measured-reference bound), ttsim timing
   (does not exist), and RTL simulation of full tiles (QuestaSim-gated for
   NEUREKA; the deterministic-dataflow closed forms already anchor to
   published cycle counts within 0.6%).

## 4. Per-workload mapping

**Qwen3-30B-A3B** (48L, 128 experts top-8, inter 768, GQA 32/4x128):
standard GQA + MoE -> ATLAS auto frontend natively; realism = memory system
+ unique-experts model + placement. Tile: Tensix-parameterized; PULP hetero
adequate at scaled RedMulE only. No index engine needed. Smallest expert
slice (2048x768) -> per-expert stream is only ~1.2 MB fp4: **row-granular
gathers, not long streams** - the calibrated 0.29-0.77 gather efficiencies
matter MORE for this model than for V4-Flash's 8.4 MB experts. This is the
model where placement experiments pay most.

**DeepSeek V4-Flash**: MoE side same as above (8.4 MB fp4 expert slices,
seq-stream friendly). Attention side needs three things ATLAS can express
and one it cannot: (a) compressed-KV gathers -> paged-KV random slots
(calibrated); (b) window attention -> short spans; (c) compressor GEMVs ->
small GEMM ops; (d) the indexer scan+top-k -> NOT a native ATLAS operator;
model as vector-engine work (pessimistic, per guardrail) or add the
HWPE-class index engine as a parameterized vector unit. The exact-DAG
analytical model remains the source of truth for op-level traffic
(validated against closed forms); ATLAS validates the memory system under
those traffic patterns.
