# Project Frontier — progress ledger

Phases follow PART XXXVI. Every phase = at least one commit on
claude/fabrik-decode-accelerator-q98dsy.

| phase | status | evidence |
|---|---|---|
| 1 Understand ATLAS | done | docs/ATLAS_ARCHITECTURE.md |
| 2 Upstream baseline | done (tilelang skipped, documented) | BASELINE.md, results/ub/ |
| 3 Source-of-truth models | done | models/source_of_truth/, model_manifest.json |
| 4 Analytical workload model | done | models/*/dag.py |
| 5 Validate byte/FLOP counts | done | tests/test_workloads.py (5/5 pass) |
| 6-7 V4/M3 in ATLAS | done via generic DAG + cycle-level e2e validation; V4/M3-specific gather patterns cycle-measured via test_dram probes | experiments/run_validation.py, calibrate_mem_eff.py |
| 8 Kernel validation | done | validation_summary.csv (0.82-1.16 ratios) |
| 9 Fabrik parameterization | done | architectures/fabrik/design_space.py |
| 10 Small DSE | done | results/frontier.csv (64,800 pts) |
| 11 Analytical vs cycle | done | fig13_validation |
| 12 Power/thermal | done | analytical/power.py, run_thermal.py, thermal_summary.csv |
| 13 Serving model | done | serving/, serving_summary.csv |
| 14 TCO model | done | economics.py, run_nre.py |
| 15 Full DSE | done | frontier.csv |
| 16 Pareto | done | results/pareto/*.csv |
| 17 Uncertainty | done | uncertainty_mc.json |
| 18 Report | in progress | report/ |

## Assumptions ledger (running)
- Memory-pattern efficiencies: RAMULATOR-MEASURED (calibrate_mem_eff.py);
  seq_stream 0.801 well-tiled (0.31 naive low case), block 0.769, random 0.287,
  scan 0.809.
- Compute-org utilizations (compute.py ORGS): ASSUMED, anchored to ATLAS
  area/power scalars; swept in MC (compute_util_scale 0.7-1.1).
- Power constants (power.py): dram 1.2 pJ/bit (0.8-2.5), iface 0.25 (0.1-0.6),
  sram 0.08, static 12% — ASSUMED, literature-typical, MC-swept.
- M3 serving precision: fp8 weights+KV base (checkpoint is bf16) — LABELED.
- MTP/speculative decoding OFF for both models — LABELED.
- GPU prefill throughput and GPU tokens/J reference — EXTERNAL ESTIMATES, labeled.
- SRAM tiling-pressure model — ASSUMED shape, anchored to measured tile-size
  sensitivity (0.20-0.80).
