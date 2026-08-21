# Fabrik: Interactive Inference for Frontier Mixture-of-Experts Models

**Fabrik Semi — Technical White Paper (draft) · August 2026**

*(Company-facing narrative draft, maintained alongside the technical report.
Numbers herein trace to PROJECT_FRONTIER_REPORT.md and
results/processed/balance_summary.json; where they diverge, the report wins.)*

---

## Summary

The inference market has split into two products that cannot be bought together. GPU serving delivers frontier mixture-of-experts (MoE) models at market economics but at per-user decode speeds of tens to low hundreds of tokens per second. SRAM dataflow machines deliver 1,000–3,000 tokens per second per user but at a capacity cost structure that excludes frontier-scale models from their public catalogs entirely. Fabrik is a proposed decode accelerator, built on hybrid-bonded 3D DRAM, targeting the region neither occupies: **1,000–2,000 output tokens/sec/user on frontier MoE models, at cost per token equal to or better than GPU serving, with materially better energy per token, at batch sizes 1–4.**

This paper states the thesis, the physics and economics behind it, and the falsifiable full-stack study — Project Frontier — now underway to validate or kill it before any RTL is written.

## 1. The gap in the inference frontier

Plot every serving option on two axes — per-user interactivity against cost per million output tokens — and a region sits empty.

GPUs anchor the economics. Frontier open MoE models serve at roughly $0.30–3 per million output tokens, but single-stream decode on a large MoE spread across eight or more devices stalls in the tens to low hundreds of tokens per second, reaching ~500 only with aggressive speculative decoding on B200-class systems. The limit is structural: HBM bandwidth divided across a tensor- and expert-parallel group, with every layer paying collective-communication and dispatch overhead.

SRAM machines anchor the interactivity records — Cerebras demonstrates 2,500 tok/s/user on Llama 4 Maverick — but on-chip weight residency costs approximately $57,000–65,000 per GB of capacity (44 GB per CS-3 wafer; 230 MB per Groq LPU). Because expert routing is data-dependent, a resident architecture must hold the *entire* model regardless of how little of it each token touches. For a 671 GB frontier MoE that implies a capital floor near $56M per model replica before the first token is served. The market evidence matches the arithmetic: public SRAM-machine rate cards top out near 120B parameters, with frontier MoE available only through custom-priced dedicated endpoints.

The empty region matters more each quarter. Reasoning models spend thousands of serial "thinking" tokens per answer; agent chains multiply serial decode through every tool call; voice interfaces impose hard real-time budgets. Sequential token generation has replaced network latency as the interactive floor of AI products. Whoever occupies this region at market economics is selling latency itself.

## 2. The workload moved toward memory

The frontier open-model ecosystem converged on two techniques that reshape the hardware problem.

**Extreme MoE sparsity.** DeepSeek V4-Flash activates ~13B of ~284B parameters per token (top-6 of 256 routed experts, FP4 expert weights, ~1M context). MiniMax M3 activates ~23B of ~428B. At ~20× sparsity, a decode step reads a small, router-selected slice of an enormous model — and at batch 1–4 there is almost nothing to amortize: under load-balanced routing, our modeling shows batch 4 touches an expected 30.5 unique experts per layer against 32 draws. Interactive decode cannot batch its way out of the bandwidth bill.

**Attention converted from a bandwidth problem into a compute problem.** Using DeepSeek-V3 (671B/37B) as the characterization vehicle, our analytical model shows weight streaming is ~86% of decode-step DRAM traffic at B=1 (routed experts alone 55%), at an arithmetic intensity of 2–8 FLOP/byte — a pure bandwidth workload. Attention has gone the other way: multi-head latent attention compresses the KV cache 57× versus plain MHA and raises the score path to ~484 FLOP/byte, and sparse-attention schemes (DeepSeek's compressed sparse attention, MiniMax Sparse Attention) cap KV reads regardless of context length. Provisioning silicon for KV bandwidth is provisioning for a problem the algorithm side is actively deleting.

What remains irreducible is expert weight bandwidth: whatever the router selects must be read, every step, from wherever it lives. That is the term Fabrik is built around.

## 3. Architecture concept

Hybrid bonding allows a compute die to be bonded face-to-face to multi-layer 3D DRAM. Hundreds of independent DRAM channels terminate directly into local compute-tile buffers over micro-bump vertical I/O measured near 0.4 pJ/bit, so card bandwidth becomes *channels × access width × frequency* rather than *pins × Gbps/pin*. This is not speculative device physics: d-Matrix's Raptor silicon measures ~105 TB/s per card at 700 MHz — roughly thirteen B300-class GPUs of bandwidth in one device — and serves as the calibration anchor for our simulation work.

Fabrik applies the substrate to exactly one job. The intended system pairs a conventional NVIDIA pool for compute-heavy prefill with Fabrik cards for token generation; a session hands off its compressed KV state once, then generates at memory speed. Each card holds the full model in stacked DRAM (128–512 GB under exploration), streams router-selected experts across 12–150 TB/s of vertical bandwidth, and executes with decode-shaped compute: reconfigurable skinny-matrix and GEMV engines at 0.25–4 PFLOPS FP4/FP8, modest SRAM, and first-class support for compressed-KV indexing and sparse-attention gather.

A governing principle from our modeling: bandwidth is provisioned per operator by arithmetic intensity, not by byte share. Expert streams — 2–8 FLOP/byte — receive the 3D-DRAM bandwidth and need only hundreds of TFLOPS to consume it. MLA-class attention at ~484 FLOP/byte is compute-provisioned and can be fed from single-digit TB/s. These are different circuits, and the card treats them differently.

## 4. Economics

| memory tier | $/GB capacity | $/(TB/s) bandwidth |
|---|---|---|
| on-chip SRAM (wafer / LPU) | ~$57,000–65,000 | ~$120–190 |
| HBM3e (B300-class GPU) | ~$156 | ~$5,600 |
| hybrid-bonded 3D DRAM (estimated) | ~$200–600 | ~$300–1,000 |

*Vendor figures are estimates from public sources; the 3D-DRAM cost column is the central uncertainty this study exists to bound.*

SRAM machines pay a capacity tax; GPUs pay a bandwidth tax; hybrid-bonded 3D DRAM is the only tier inexpensive on both axes simultaneously — which is precisely the combination a 20×-sparse MoE at low batch demands: hold everything, stream a slice.

Modeled at a fixed 2,000 tok/s/user on a 671 GB-class MoE, matching B300-class cost per token permits a Fabrik card cost of up to roughly $70k at B=1, rising to ~$560k at B=16 — under the deliberately conservative assumption that GPUs scale perfectly across the eleven-plus devices the bandwidth arithmetic requires of them. Measured GPU behavior at low batch on frontier MoE falls one to two orders of magnitude below that roofline, lost to collectives, cross-device expert dispatch, and launch overhead. That measured gap is Fabrik's economic margin, and producing a defensible number for it is a primary output of the study.

## 5. Project Frontier: validate before silicon

The feasibility study is built on ATLAS (MICRO '26), the first silicon-proven full-stack simulator for 3D-DRAM LLM accelerators (validated to ≤8.6% error against fabricated hardware), integrating Ramulator DRAM timing, BookSim network modeling, and HotSpot thermal analysis in the performance loop. An independent millisecond-scale analytical model cross-checks every cycle-level result; a Raptor calibration case anchors the memory model to published silicon; workload models for V4-Flash and M3 are built from source-of-truth configurations with sparse attention modeled explicitly rather than proxied by dense attention; expert routing is trace-driven, not assumed; and a serving-level simulator reports tail interactivity (p95/p99 TPOT) under continuous batching rather than isolated microbenchmarks. The economics layer covers TCO and custom-silicon NRE amortization across fleet sizes.

The study deliberately inverts the usual question. It does not ask how fast 100 TB/s is; it asks the *minimum* bandwidth, compute, SRAM, and power that achieve 1,000 / 1,500 / 2,000 tok/s/user at batch 1–4 across 32K–1M context, and what that hardware costs to build and operate within a 400–1,000 W thermal envelope.

Falsification criteria are pre-registered. The thesis is flagged as unsupported if usable bandwidth at B=1–4 falls below ~25% of peak, if more than ~4–5 PFLOPS is required merely to consume the memory system, if interactivity targets demand over 1 kW or thermally unsustainable operating points, if package and capacity economics dominate TCO, or if modeled cost per token remains materially worse than optimized GPU serving. A negative result ends the project before RTL. That is the point of doing it this way.

## 6. Deliverable

One chart, two views: per-user tokens/sec against dollars per million output tokens, and against tokens per joule. Either a physically plausible Fabrik configuration occupies the region no shipping system reaches — with its capacity, bandwidth, compute organization, power, and cost quantified, with uncertainty bounds — or the specific physical or economic bottleneck that forbids it is identified and the capital is saved.

**Status.** As of this draft the study has delivered its first full pass: see `PROJECT_FRONTIER_REPORT.md` for the verdict (SUPPORTED for V4-Flash-class; PLAUSIBLE BUT UNPROVEN for M3-class), the 64,800-point DSE, cycle-level validation, thermal results, and the DeepSeek-V3 addendum quantifying why compressed-attention models are the design target. *Contact: [—]*
