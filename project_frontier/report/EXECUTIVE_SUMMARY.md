# Project Frontier — Executive Summary

**Question: can Fabrik — a purpose-built decode accelerator on 3D DRAM —
deliver Cerebras-level interactivity at NVIDIA-level cost?**

**Answer: yes for the workload class it should be built for.** For
DeepSeek-V4-Flash-class models (compressed-attention MoE, ~13B active
parameters), a moderate Fabrik configuration — **50 TB/s of 3D-DRAM
bandwidth, 1 PFLOP/s FP8, 256 GB, 64 MB SRAM, ~480–560 W** — simultaneously
delivers, per our cycle-validated analytical model:

- **2,560 tokens/s/user at batch 1** and **1,480 at batch 4** (128K context;
  2,180 at 1M) — 8–20× the 107–306 tok/s users get from measured GPU serving
  of the same model today, and in the band Cerebras publishes for much
  smaller dense models;
- **$0.16/1M output tokens at B=1 and $0.07 at B=4** (at $25K/system, 60%
  fleet utilization) vs $0.168–0.66/1M across today's same-model GPU routes —
  NVIDIA-class or better economics, with $50M–400M of silicon NRE amortizing
  by 1k–10k deployed units;
- **5–11 tokens/J**, roughly 5–10× our estimate for GPU serving at iso-model.

This holds up under scrutiny we designed to break it: the analytical model
matches ATLAS cycle simulation within ±20% on dense and MoE models with zero
tuning; every memory access pattern is priced at Ramulator-measured
efficiency (sequential streams 0.80 of peak, V4's scattered index-gathers
only 0.29 — modeled, not wished away); HotSpot says the flagship runs at
84–92 °C on liquid cooling; and a 2,000-sample Monte Carlo over every assumed
constant keeps batch-1 throughput above 2,100 tok/s at the 5th percentile.
The physics is simple and robust: V4-Flash moves ~11 GB per token; 50 TB/s
at measured efficiencies turns that into ~0.4 ms per token, and at batch ≤4
almost nothing else matters. Expert weights do not amortize at batch 1–4
(3% overlap at B=4) — which is precisely why a bandwidth-first machine beats
a compute-first one here, and why GPUs must run big batches that destroy
per-user speed.

**The important nuance is workload dependence.** MiniMax M3 (428B/23B-active,
MSA sparse attention) is a harder fit: it moves 26 GB/token (fp8), so 1,000
tok/s/user needs 40–100 TB/s and 2,000 needs ~100 TB/s at ~1 kW — grazing our
falsification threshold — and its 428 GB fp8 weights + 64 GB/user of 1M KV
make **capacity, not bandwidth, the binding constraint**: batch 4 at 1M
context does not fit even 512 GB on one device. M3 works at batch 1 with
better-than-GPU economics, but the full thesis is only *plausible* there, and
would need multi-device Fabrik or fp4 weights to close. A second nuance:
V4-Flash's Lightning-Indexer scan is the one place specialized hardware
clearly pays (a scan/top-k engine is worth ~1.6× at 1M context); M3's MSA
runs fine on commodity engines.

**What we would build:** the 50 TB/s / 1 PFLOP/s / 256 GB / 64 MB "flagship"
— 49 compute tiles, each on its own 1 TB/s DRAM vault (the configuration
class ATLAS itself validates), hybrid GEMV+reconfigurable-systolic compute
with a dedicated index engine, in a 500–600 W liquid-cooled package. On the
two charts that matter — TPS/user vs $/1M tokens, and TPS/user vs tokens/J —
this point sits in a region **no GPU offering of these models and no
published wafer-scale offering occupies**: >2,000 tok/s/user at ≤$0.17/1M and
>5 tok/J. The margin at batch 1 is ~8× on interactivity at equal cost, and
the Monte-Carlo p5 keeps >2,100 tok/s. The risks that matter, in order:
real silicon cost per system, achievable fleet utilization, sustained
streaming efficiency, DRAM array energy, and compute utilization — all
enumerated with ranges in the report. None of them, at their pessimistic
bounds, pushes the V4-class result out of the claimed region.

**Verdict: SUPPORTED for V4-Flash-class decode; PLAUSIBLE BUT UNPROVEN for
M3-class. Proceed to RTL-level power/area validation of the flagship point,
with V4-Flash-class models as the design target and multi-device capacity
scaling as the M3 contingency.**
