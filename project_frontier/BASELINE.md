# Upstream Baseline (Part II)

## Environment record

| item | value |
|---|---|
| ATLAS commit | b278739 (tibrewalrachit/ATLAS-MICRO-2026, fork of pku-gsun) |
| ramulator2 submodule | e442c64 + ATLAS patch series (patches/ramulator2.patch -> d6d2dfe) |
| booksim2 submodule | 28f4329 + ATLAS patch series (patches/booksim2.patch -> 437507f) |
| pybind11 / yaml-cpp | 4f81a12 / a83cd31 |
| HotSpot | f18831e (built with libblas-dev, libsuperlu-dev) |
| local host | 4-core x86_64 container, 15 GB RAM, Linux 6.18.44 |
| remote (heavy sims) | Modal cloud workers, 32 vCPU / 64 GB, image = this patched tree |
| compiler | g++ 13.3.0, CMake 3.28.3 (local); Debian slim + same toolchain (Modal) |
| Python | 3.11.15 (upstream recommends 3.10; no incompatibilities encountered) |
| key Python deps | numpy 2.4.6, pandas 3.0.5, pyyaml, z3-solver, loguru, transformers 4.51.0 |

## Build commands (exact)

```bash
git submodule update --init simulator/3rd/pybind11 simulator/3rd/yaml-cpp \
    simulator/src/dram/ramulator2 simulator/src/noc/booksim2 pyta/thirdparty/hotspot
bash patches/unpatch_booksim2.sh && bash patches/unpatch_ramulator2.sh
python3 -m venv /home/user/venv-atlas && . /home/user/venv-atlas/bin/activate
pip install "cmake>=3.26.1" ninja cython scikit-build-core "patchelf>=0.17.2" \
    z3-solver loguru matplotlib numpy pandas psutil pyyaml scipy transformers==4.51.0
cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel 4
pip install -e simulator --no-build-isolation
apt-get install -y libblas-dev libsuperlu-dev && bash pyta/build_hotspot.sh
```

## Documented deviations from upstream instructions

1. **TileLang frontend skipped** (`tests/inference_test_tilelang.py` not run):
   it requires building the TVM submodule (hours, GPU-oriented); the auto and
   ATLang frontends fully cover this project's needs. No upstream source or
   expected results were modified.
2. **Heavy experiments run on Modal** (32-core cloud workers) rather than the
   4-core dev container; the reference AE machine is 96 cores. Same code, same
   configs; wall-clock only differs.
3. **Reduced DSE slice**: one full cloud DRAM-DSE experiment family
   (`--experiment io`) was run rather than all four families (full AE = 12.4 h
   on 96 cores). Runtime on Modal: 164 s.

## Kick-the-tires results (cycle-simulated, unmodified upstream)

| frontend | system | model | batch | ctx | e2e latency (s) | e2e energy (J) |
|---|---|---|---|---|---|---|
| auto | cloud | opt_66b | 8 | 4096 | 0.0025443 | 0.30566 |
| auto | edge | opt_6.7b | 8 | 2048 | 0.0132080 | 1.04873 |
| atlang | cloud | mixtral_8x22b | 8 | 16384 | 0.0023996 | 0.28829 |
| atlang | edge | palm_8b | 8 | 2048 | 0.0094979 | 0.75043 |

MoE auto-frontend run (mixtral_8x22b, bs4, ctx4096) and the 5-case validation
matrix are recorded in `results/processed/validation_summary.csv` when the
Modal jobs complete; raw CSVs under `results/ub/` (from the shared
`frontier-results` Modal volume, prefix `upstream_baseline/`).

Cloud DRAM DSE (io family) outputs: `results/ub/upstream_baseline/dram_dse_io/`
including `fig10b_io_organization.{png,pdf}` — trends match
`docs/ref_figs/fig10b_io_organization.png` (upstream notes absolute values may
differ due to parameter anonymization).

No upstream expected results were modified.
