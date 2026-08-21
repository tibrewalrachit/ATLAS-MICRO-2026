"""Reproduce the full analytical pipeline from a fresh clone.

Usage: python experiments/run_all.py [--analytical-only]
Cycle-level calibration/validation artifacts (results/processed/*.json/csv
from Ramulator/ATLAS/HotSpot runs) are committed; pass nothing to reuse them.
Rerunning them requires the ATLAS native build (see ../BASELINE.md).
"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["run_dse.py", "run_pareto.py", "run_min_hw.py", "run_nre.py",
         "run_orgs.py", "run_disagg.py", "run_serving.py",
         "run_uncertainty.py", "make_figures.py"]
for s in STEPS:
    print(f"=== {s}")
    r = subprocess.run([sys.executable, os.path.join(HERE, s)])
    if r.returncode != 0:
        sys.exit(f"step {s} failed")
print("all steps complete; see results/")
