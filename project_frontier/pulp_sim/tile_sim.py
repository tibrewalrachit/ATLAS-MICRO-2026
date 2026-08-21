"""PULP tile simulator: compose per-kernel cycles from HWPE closed-form
models + TCDM contention + Snitch orchestration overhead.

Fidelity labels per component:
  - HWPE busy cycles: closed-form, paper-anchored (redmule_model, neureka_model)
  - TCDM stalls: pessimistic analytical (tcdm)
  - Orchestration: Banshee-measured constants from
    results/processed/banshee_orchestration.json when present, else ASSUMED
    defaults (ranged) - the file records which was used.
DRAM feed is NOT modeled here (die layer / ATLAS owns it); tile_sim reports
the compute-side cycle count and the bytes/clk the tile demands, so the die
composer can take max(tile_time, memory_time) per operator.
"""
import json, os, math
import redmule_model as rm
import neureka_model as nk
import tcdm

_ORCH_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                          "processed", "banshee_orchestration.json")
_DEFAULT_ORCH = {"hwpe_job_setup_cycles": 120, "dma_setup_cycles": 60,
                 "ssr_loop_setup_cycles": 40, "source": "ASSUMED (banshee not yet run)"}


def orchestration():
    try:
        return json.load(open(_ORCH_PATH))
    except Exception:
        return dict(_DEFAULT_ORCH)


def expert_gemv(K, N, batch=1, engine="neureka", qw=4, freq_ghz=1.0):
    """One expert GEMV slice on one tile. Returns cycles, seconds, and the
    DRAM bytes/clk demand while busy (for die-level max())."""
    o = orchestration()
    if engine == "neureka":
        busy = nk.gemv_cycles(K, N, batch, qw)
        wbytes = K * N * qw / 8
        hwpe_bpc = wbytes / busy
    else:  # redmule fp16
        busy = rm.gemm_cycles(batch, N, K)
        wbytes = K * N * 2
        hwpe_bpc = wbytes / busy
    dma_bpc = hwpe_bpc                     # double-buffered fill at same rate
    stall = tcdm.stall_factor(hwpe_bpc, dma_bpc)
    cycles = busy * stall + o["hwpe_job_setup_cycles"] + o["dma_setup_cycles"]
    return {"cycles": cycles, "seconds": cycles / (freq_ghz * 1e9),
            "weight_bytes": wbytes, "demand_bytes_per_clk": dma_bpc,
            "tcdm_stall_factor": stall,
            "orchestration_source": o.get("source", "banshee")}


def attention_chunk(heads, head_dim, span, batch=1, freq_ghz=1.0):
    """Attention score+AV on RedMulE fp16 (per tile share)."""
    o = orchestration()
    busy = rm.gemm_cycles(batch * heads, span, head_dim) + \
           rm.gemm_cycles(batch * heads, head_dim, span)
    cycles = busy + 2 * o["hwpe_job_setup_cycles"]
    return {"cycles": cycles, "seconds": cycles / (freq_ghz * 1e9)}
