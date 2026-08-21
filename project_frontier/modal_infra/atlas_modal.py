"""Modal infrastructure for running ATLAS simulations on large cloud workers.

The local dev container has 4 cores; ATLAS DSE/cycle simulations are CPU-hungry
(reference machine: 96 cores). This module bakes the locally patched ATLAS tree
into a Modal image, builds the native simulator once (cached), and exposes a
generic `run_atlas` function that executes an arbitrary command inside the repo
and returns stdout/stderr plus any requested output files.

Usage:
    modal run project_frontier/modal_infra/atlas_modal.py --cmd "python tests/inference_test_auto.py"
or import `run_atlas` from other drivers via modal.Function.from_name.
"""

import modal

REPO_ROOT = "/root/atlas"

# The image bakes the *patched* working tree (submodules already patched with
# ATLAS's patch series locally) and rebuilds the native simulator remotely.
atlas_source = modal.Image.debian_slim(python_version="3.11").apt_install(
    "build-essential", "cmake", "ninja-build", "git", "flex", "bison",
    "libblas-dev", "liblapack-dev", "libsuperlu-dev", "wget"
).pip_install(
    "cmake>=3.26.1", "ninja", "cython", "scikit-build-core", "patchelf>=0.17.2",
    "z3-solver", "loguru", "matplotlib", "numpy", "pandas", "psutil", "pyyaml",
    "scipy", "transformers==4.51.0",
).add_local_dir(
    "/home/user/ATLAS-MICRO-2026",
    remote_path=REPO_ROOT,
    copy=True,
    ignore=[".git", "**/.git", "simulator/build", "kick_the_tires", "results",
            "project_frontier", "**/__pycache__"],
).run_commands(
    # Native build: simulator (atlasim + test_dram) and HotSpot for thermal runs
    f"cd {REPO_ROOT} && cmake -S simulator -B simulator/build -DCMAKE_BUILD_TYPE=Release"
    f" && cmake --build simulator/build --parallel $(nproc)"
    f" && python -m pip install -e simulator --no-build-isolation",
    f"cd {REPO_ROOT}/pyta/thirdparty/hotspot && make -j$(nproc) || (echo 'hotspot build failed (non-fatal for perf-only runs)'; true)",
)

app = modal.App("frontier-atlas")
results_vol = modal.Volume.from_name("frontier-results", create_if_missing=True)

# 30 min default; DSE cases override via run_atlas.spawn options at call sites.
@app.function(image=atlas_source, cpu=32, memory=65536, timeout=60 * 60 * 6,
              volumes={"/results": results_vol})
def run_atlas(cmd: str, collect: list[str] = None, collect_to: str = None,
              env: dict = None, timeout_s: int = 21000) -> dict:
    """Run `cmd` from the ATLAS repo root; return stdout/stderr/rc.

    collect: glob patterns (relative to repo root) of output files to persist.
    collect_to: subdirectory under the shared `frontier-results` volume.
    """
    import subprocess, os, glob, shutil, time
    t0 = time.time()
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run(cmd, shell=True, cwd=REPO_ROOT, env=e,
                       capture_output=True, text=True, timeout=timeout_s)
    saved = []
    if collect and collect_to:
        dst_root = os.path.join("/results", collect_to)
        for pattern in collect:
            for src in glob.glob(os.path.join(REPO_ROOT, pattern), recursive=True):
                rel = os.path.relpath(src, REPO_ROOT)
                dst = os.path.join(dst_root, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    saved.append(rel)
        results_vol.commit()
    return {"rc": p.returncode, "stdout": p.stdout[-200_000:], "stderr": p.stderr[-100_000:],
            "wall_s": round(time.time() - t0, 1), "saved": saved, "nproc": os.cpu_count()}


@app.local_entrypoint()
def main(cmd: str = "python -c 'import atlasim; print(atlasim.__file__)'",
         collect: str = "", collect_to: str = "", cpu: int = 32):
    r = run_atlas.remote(cmd, collect.split(",") if collect else None, collect_to or None)
    print(f"rc={r['rc']} wall={r['wall_s']}s nproc={r['nproc']}")
    print(r["stdout"][-8000:])
    if r["rc"] != 0:
        print("STDERR:", r["stderr"][-4000:])
