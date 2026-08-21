"""Core analytical decode-step model (Parts VI, IX).

decode_step() composes a workload DAG (per-token B=1 basis), a batch size,
an expert-reuse model, and a Fabrik hardware point into decode-step latency,
per-category time/traffic decomposition, TPS/user, and utilizations.

Timing model per operator: T_op = max(T_mem, T_comp) (+ fixed op latency),
i.e. weights/KV stream while compute proceeds (double buffering); operators
are dependency-serialized, so step time = sum over ops. A global lower bound
max(sum T_mem, sum T_comp) is also reported; base results use the per-op
model, and the gap between the two is the pipelining uncertainty band.
"""
import sys, os
from dataclasses import dataclass, field
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from memory import MemorySystem
from compute import ORGS, ComputeOrg, op_util
import moe as moe_mod


@dataclass
class FabrikPoint:
    name: str
    peak_bw_TBps: float
    peak_pflops_fp8: float
    capacity_GB: float
    sram_MB: float
    org: str = "hybrid"
    fixed_op_latency_s: float = 2e-7      # kernel issue/sync per op instance chain
    per_layer_sync_s: float = 3e-7        # NoC/reduction sync per layer
    mem_eff: dict = None                  # override pattern efficiencies

    def memory(self) -> MemorySystem:
        ms = MemorySystem(self.peak_bw_TBps * 1e12, self.capacity_GB * 1e9)
        if self.mem_eff:
            ms.eff.update(self.mem_eff)
        # SRAM effect: streaming efficiency needs double-buffered weight/act
        # tiles; below ~16 MB per TB/s-class stream the tiling explorer is
        # forced into small tiles -> extra row activations and re-reads.
        # Calibration anchor: test_dram matrix probes (tile 512x8192 -> 0.80,
        # small tiles -> 0.20-0.31). Modeled as a smooth penalty, labeled
        # assumption in report.
        req_mb = 16.0 + 2.0 * self.peak_bw_TBps / 10.0
        f = min(1.0, 0.55 + 0.45 * min(1.0, self.sram_MB / req_mb))
        ms.eff = {k: (v * f if k != "on_chip" else v) for k, v in ms.eff.items()}
        return ms


@dataclass
class StepResult:
    latency_s: float
    latency_lower_s: float
    tps_per_user: float
    aggregate_tps: float
    by_category: dict           # cat -> dict(t_mem, t_comp, t, bytes, flops)
    bytes_per_step: float
    bytes_per_token: float
    flops_per_token: float
    dram_util: float            # achieved/peak while streaming
    compute_util: float
    ai: float
    kv_capacity_per_user: float = 0.0
    fits_capacity: bool = True


def _expert_scale(n_experts, top_k, batch, reuse_model, zipf_alpha=1.0):
    """Weight-stream multiplier for routed experts relative to B=1 stream."""
    if reuse_model == "none":
        u = moe_mod.unique_experts_none(n_experts, top_k, batch)
    elif reuse_model == "statistical":
        u = moe_mod.unique_experts_statistical(n_experts, top_k, batch)
    elif reuse_model == "statistical_zipf":
        u = moe_mod.unique_experts_statistical(
            n_experts, top_k, batch, moe_mod.zipf_probs(n_experts, top_k, zipf_alpha))
    else:
        raise ValueError(reuse_model)
    return u / top_k     # relative to one user's top_k stream


MOE_SHAPE = {"deepseek_v4_flash": (256, 6), "minimax_m3": (128, 4)}


def decode_step(workload, hw: FabrikPoint, batch: int = 1,
                reuse_model: str = "statistical") -> StepResult:
    mem = hw.memory()
    org = ORGS[hw.org]
    peak_flops = hw.peak_pflops_fp8 * 1e15
    n_exp, top_k = MOE_SHAPE[workload.model]
    exp_scale = _expert_scale(n_exp, top_k, batch, reuse_model)

    by_cat = {}
    n_op_instances = 0
    tot_mem_t = tot_comp_t = 0.0
    tot_bytes = tot_flops = 0.0
    for op in workload.ops:
        n_op_instances += op.count
        # ---- bytes per decode STEP (batch-composed) ----
        if op.category == "routed_expert":
            wb = op.weight_bytes * exp_scale          # unique experts streamed once
        elif op.shared_across_batch:
            wb = op.weight_bytes                      # one stream serves all users
        else:
            wb = op.weight_bytes * batch
        per_user = op.kv_read_bytes + op.kv_write_bytes + op.index_read_bytes + op.act_bytes
        step_bytes = (wb + per_user * batch) * op.count
        # ---- FLOPs per step: every user computes ----
        fp4 = op.precision == "fp4" and org.fp4_double
        eff_peak = peak_flops * (2.0 if fp4 else 1.0)
        util = op_util(org, op.engine.value, op.category, batch)
        step_flops = op.flops * batch * op.count
        t_mem = mem.time_for({op.pattern.value: step_bytes})
        t_comp = step_flops / (eff_peak * util) if step_flops else 0.0
        c = by_cat.setdefault(op.category, dict(t_mem=0.0, t_comp=0.0, t=0.0,
                                                bytes=0.0, flops=0.0))
        c["t_mem"] += t_mem; c["t_comp"] += t_comp
        c["t"] += max(t_mem, t_comp)
        c["bytes"] += step_bytes; c["flops"] += step_flops
        tot_mem_t += t_mem; tot_comp_t += t_comp
        tot_bytes += step_bytes; tot_flops += step_flops

    fixed = hw.fixed_op_latency_s * min(n_op_instances, 400) + hw.per_layer_sync_s * 60
    latency = sum(c["t"] for c in by_cat.values()) + fixed
    latency_lower = max(tot_mem_t, tot_comp_t) + fixed
    tps = 1.0 / latency

    # capacity check (weights + batch x KV). Workloads may carry explicit
    # weight_capacity_bytes / kv_per_user_bytes attributes; the two original
    # models fall back to their modules; unknown models skip the check.
    kv_cap = getattr(workload, "kv_per_user_bytes", None)
    wt_bytes = getattr(workload, "weight_capacity_bytes", None)
    if wt_bytes is None:
        if workload.model == "deepseek_v4_flash":
            from deepseek_v4_flash.dag import kv_capacity_per_user, total_params
            kv_cap = kv_capacity_per_user(workload.context)
            wt_bytes = total_params()["total"] * 0.75   # mixed fp4/fp8 avg
        elif workload.model == "minimax_m3":
            from minimax_m3.dag import kv_capacity_per_user, total_params
            kp = 1.0 if workload.mode.endswith("fp8") else 2.0
            kv_cap = kv_capacity_per_user(workload.context, kp)
            wt_bytes = total_params()["total"] * kp
    if wt_bytes is not None:
        fits = wt_bytes + batch * (kv_cap or 0.0) <= hw.capacity_GB * 1e9
    else:
        fits = True

    return StepResult(
        latency_s=latency, latency_lower_s=latency_lower,
        tps_per_user=tps, aggregate_tps=tps * batch,
        by_category=by_cat, bytes_per_step=tot_bytes,
        bytes_per_token=tot_bytes / batch, flops_per_token=tot_flops / batch,
        dram_util=(tot_bytes / latency) / mem.peak_bw,
        compute_util=(tot_flops / latency) / peak_flops,
        ai=tot_flops / tot_bytes, kv_capacity_per_user=kv_cap, fits_capacity=fits)
