# Project Frontier

Full-stack feasibility study of **Fabrik**, a 3D-DRAM decode accelerator, on
the ATLAS (MICRO '26) simulation infrastructure.

- **Question & verdict:** `report/EXECUTIVE_SUMMARY.md` (2 pages) and
  `report/PROJECT_FRONTIER_REPORT.md` (full 22-section report).
- **Reproduce analytics + figures:** `pip install -r requirements.txt &&
  python experiments/run_all.py` (seconds-to-minutes; uses committed
  calibration artifacts).
- **Reproduce cycle-level artifacts:** build ATLAS per `BASELINE.md`, then
  `experiments/calibrate_mem_eff.py` (Ramulator), `run_validation.py`
  (vs ATLAS cycle sim), `run_thermal.py` (HotSpot),
  `calibration/raptor/raptor_calibration.py`.
- **Model sources of truth:** `models/source_of_truth/` +
  `models/model_manifest.json` (per-parameter provenance/confidence).
- **Heavy runs on Modal:** `modal_infra/atlas_modal.py` (deployed app
  `frontier-atlas`; results volume `frontier-results`).

Layout follows the project brief (analytical/, models/, architectures/fabrik/,
calibration/raptor/, serving/, economics/, baselines/, experiments/, results/,
docs/, report/). `docs/PROGRESS.md` is the phase + assumptions ledger.
