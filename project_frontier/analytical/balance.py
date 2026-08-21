"""Card balance & per-operator bandwidth provisioning (FabrikSim §3, §6, §7).

Three analyses:
1. demanded_balance(): workload-demanded GB of capacity per TB/s of bandwidth
   at a target TPS/user — the ratio that decides whether a card's
   capacity:bandwidth split matches the operating regime. A card's own ratio
   (capacity_GB / peak_TBps) should sit near the demanded ratio; far above it
   the bandwidth is stranded, far below the capacity is.
2. operator_bandwidth_split(): provision bandwidth per operator by arithmetic
   intensity, not byte share — returns, for a given compute budget, how much
   bandwidth each operator class can actually consume
   (BW_consumable = compute / AI), exposing the MLA result: at 2 PFLOP/s,
   484-FLOP/byte attention can consume only ~4 TB/s.
3. machine_balance(): FLOP/byte machine ratios for reference machines vs the
   workload's operator intensities (memory- vs compute-bound per operator).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import moe as moe_mod


def demanded_balance(workload, weight_capacity_gb, kv_per_user_gb,
                     batch, target_tps, reuse=("statistical",)):
    """GB of required capacity per TB/s of required bandwidth."""
    s = workload.summary()
    from model import MOE_SHAPE, _expert_scale
    n_exp, top_k = MOE_SHAPE.get(workload.model, (256, 8))
    esc = _expert_scale(n_exp, top_k, batch, reuse[0])
    t = workload.totals()
    exp_b = t.get("routed_expert", {"weight": 0})["weight"]
    shared_stream = s["weight"] - exp_b                     # amortizes over batch
    per_user = s["kv_r"] + s["kv_w"] + s["idx"] + s["act"]
    step_bytes = shared_stream + exp_b * esc + per_user * batch
    bw_needed_tbps = step_bytes * target_tps / 1e12         # per decode step rate
    cap_needed_gb = weight_capacity_gb + batch * kv_per_user_gb
    return {"bw_needed_TBps": bw_needed_tbps,
            "capacity_needed_GB": cap_needed_gb,
            "gb_per_tbps": cap_needed_gb / bw_needed_tbps}


def operator_bandwidth_split(workload, compute_pflops, batch=1):
    """Per operator class: bytes/step, AI, and the bandwidth that the given
    compute budget could actually consume for it (compute/AI)."""
    out = {}
    t = workload.totals()
    for cat, c in t.items():
        byts = c["weight"] + c["kv_r"] + c["kv_w"] + c["idx"] + c["act"]
        if byts <= 0:
            continue
        ai = c["flops"] / byts
        out[cat] = {"bytes_per_step_gb": byts / 1e9, "flop_per_byte": ai,
                    "bw_consumable_tbps": (compute_pflops * 1e15 / ai) / 1e12
                                          if ai > 0 else float("inf")}
    return out


REF_MACHINES = {   # peak DENSE FP8 FLOP/s over memory BW (public spec sheets;
    # sparse/FP4 marketing numbers excluded on purpose)
    "B300": 4.5e15 / 8e12,         # ~4.5 PF dense fp8 / 8 TB/s -> 562
    "H200": 2e15 / 4.8e12,         # ~2 PF fp8 / 4.8 TB/s -> 417
    "fabrik_2pf_105tbps": 2e15 / 105e12,   # 19: compute-lean by design
}
