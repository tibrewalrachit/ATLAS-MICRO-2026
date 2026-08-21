# ATLAS Architecture Notes (Project Frontier, Part I)

This document records how ATLAS works end-to-end, established by direct code
inspection of this repository (commit `b278739`, fork of pku-gsun/ATLAS-MICRO-2026),
and identifies which components Project Frontier reuses unchanged vs. extends
for the Fabrik decode-accelerator study.

## 1. End-to-end pipeline

```
model config JSON (configs/models/*.json, HF-config style)
      │  frontend/model_parser.py
      │    get_model_config_from_hf() -> ModelConfig
      │    get_layer_operator_list()  -> attention_block + ffn_moe_block
      │        Operator{GEMM(B,M,K,N) | ATTENTION(kv_heads,groups,head_dim,ctx)
      │                 | VECTOR | ALLREDUCE | ALL2ALL}, TP/EP-sharded shapes
      ▼
frontend/auto/system.py  (CloudSystem / EdgeSystem)
      │    explore_cloud_tiling()  (frontend/auto/tiling_explorer_cloud.py)
      │      - multiprocess DSE over per-operator tilings
      │      - emits per-op task descriptions + per-core placement
      ▼
YAML contract consumed by the C++ simulator:
      data_placement.yaml        (tensor -> channel/core placement)
      operator_description.yaml  (ordered op list w/ types, tiles, deps)
      attn_input.yaml            (decode attention: paged KV slot map per core;
                                  supports `random` slot assignment)
      ▼
atlasim.Chip(arch_yaml, op_yaml, placement_yaml).simulate()   [pybind11 -> C++]
      simulator/src/
        core/core_array: N cores, each {controller, matrix_unit, vector_unit,
                          buffer (SRAM, size + rd/wr BW), noc_interface}
        matrix_unit: cycles = ceil(MACs / mac_num)     <- shape-agnostic!
        noc/noc_wrapper: BookSim2 (patched), mesh, flit_size
        dram/dram_wrapper: Ramulator2 (patched) "HBDRAM" device
            org presets:  HBDRAM_<density>Gb_<pins>pin_<cols>col
            timing preset: HBDRAM_500Mbps (500 Mbps/pin)
            e.g. 16 ch x 1024 pin x 500 Mbps = 1 TB/s   (cloud_1TBps.yaml)
      ▼
Stats { e2e_cycles, matrix/vector/buffer/dram/noc cycles,
        per-component energy (J), flop_count, memory_access_bytes,
        matrix/vector/buffer/dram utilization }
      ▼
frontend adds: inter-chip comm (analytical: BW + latency + pJ/bit),
               per-layer scaling (x num_layers), energy = component power x time
      ▼
pyta/ (PyTA + HotSpot): thermal_evaluator(dims, layers, cooling, logic/DRAM power)
      -> max temperature + thermally sustainable frequency_MHz
      -> fed back into the chip YAML frequency before performance runs
```

Key file references:
- `frontend/model_parser.py` — ModelConfig covers GQA/MHA/MQA, Mixtral/Qwen MoE,
  DeepSeek-V3-style MLA. Single global `element_size` (bytes/element).
- `frontend/hardware_parser.py` — ChipConfig {frequency, core_num, matrix.mac_num,
  vector.vec_num, buffer.{size,read_bw,write_bw}, dram.config_path, noc.*}; every
  component carries `power` (W) and `area` (mm^2) used for energy/area accounting.
- `configs/architecture/system/*.yaml` — CloudSystemConfig adds tp/ep size+scope,
  interconnect (scale-up/out BW, latency, pJ/bit), paged-KV block size.
- `simulator/src/common/stats.h` — the Stats contract above.
- `docs/ae_instructions.md` — 4 DSE families; 12.4 h on 96 cores (we run heavy
  jobs on Modal workers instead; see `project_frontier/modal_infra/`).

## 2. Component observations relevant to Fabrik

1. **Matrix engine is a MAC-throughput model** (`ceil(MACs/mac_num)`): skinny
   GEMV shapes do not intrinsically lose utilization inside the engine; shape
   effects arise via tiling, SRAM feed BW, and DRAM behavior. Consequence: the
   Part XIII compute-organization study (systolic vs GEMV vs hybrid) must model
   shape-dependent utilization in our analytical layer, with ATLAS supplying
   memory-system-limited feed rates.
2. **HBDRAM is the 3D-DRAM model**: per-channel pins x rate gives per-channel
   BW; channel count scales aggregate BW. Fabrik configs (12–150 TB/s) are
   generated as new org presets / channel counts + Ramulator YAMLs.
3. **Decode attention uses paged KV slot maps** incl. randomized slot
   assignment — a direct hook for modeling sparse gathers (CSA top-512 rows,
   MSA 128-token blocks) as slot patterns.
4. **Energy model** = per-component power (from chip YAML) x active cycles +
   Ramulator DRAM energy; interconnect at pJ/bit. Power/area numbers for the
   paper's cloud chip are in `configs/architecture/chip/cloud/stratum/atlas.yaml`.
5. **AE caveat**: upstream states architecture parameters are anonymized and
   some microarchitecture simplified for an industry collaborator — absolute
   values are trend-faithful, not silicon-calibrated. Project Frontier therefore
   cross-checks with an independent analytical model + a Raptor-like
   calibration case rather than trusting any single layer.

## 3. Reuse vs. extend

| Component | Status for Fabrik |
|---|---|
| Ramulator2 HBDRAM device + timing/org presets | **Reuse**, add Fabrik-scale presets (channel count/pins/rate/row size) |
| Cycle simulator core (cores/SRAM/NoC/DRAM) | **Reuse unchanged** |
| BookSim NoC integration | **Reuse unchanged** (larger meshes via config) |
| PyTA/HotSpot thermal flow | **Reuse unchanged** (Fabrik floorplans as inputs) |
| Auto frontend for standard GQA/MoE models | **Reuse** for upstream baselines only |
| `model_parser.py` operator builder | **Bypass** for V4-Flash/M3: their CSA/HCA/Indexer and MSA operators don't fit; we generate `operator_description.yaml`/placement YAML directly from our own DAG builders (`project_frontier/models/*`) |
| Global `element_size` | **Extend**: per-operator precision (FP4 experts / FP8 attention / fp32 compressor) handled in our DAG generator by scaling bytes per op |
| Batch-dependent expert reuse | **New** (analytical layer, Part VII) |
| Analytical roofline simulator | **New** (`project_frontier/analytical/`) |
| Serving / TCO / NRE / baselines | **New** (`project_frontier/{serving,economics,baselines}/`) |
| Heavy-run execution | **New**: Modal workers (32 cpu) via `project_frontier/modal_infra/atlas_modal.py`; local 4-core container does dev + analytical sweeps |

## 4. Baseline & environment record

See `BASELINE.md` (Part II) for exact versions, commands, and outputs.
