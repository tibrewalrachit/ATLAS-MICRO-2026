"""Part XX: persistent kernel/decode-step result cache.

Key: (model, mode, op-or-step, batch, context, precision-tag, arch signature,
mapping/org). Values: latency, bytes, per-engine utilization, energy, and the
simulation method that produced them (analytical | atlas_cycle). Cycle-level
entries are written by the validation pipeline; analytical entries on demand.
Composition of full-model decode from cached per-step entries is what the
serving simulator uses (ServingSim hits this cache through step_latency).
"""
import json, os, threading

_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "processed",
                     "kernel_cache.json")
_LOCK = threading.Lock()


def _load():
    try:
        return json.load(open(_PATH))
    except Exception:
        return {}


def _key(model, mode, step, batch, context, arch_name, org):
    return "|".join(map(str, (model, mode, step, batch, context, arch_name, org)))


def get(model, mode, step, batch, context, arch_name, org):
    return _load().get(_key(model, mode, step, batch, context, arch_name, org))


def put(model, mode, step, batch, context, arch_name, org, value: dict):
    with _LOCK:
        d = _load()
        d[_key(model, mode, step, batch, context, arch_name, org)] = value
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        json.dump(d, open(_PATH, "w"), indent=1)


def cached_decode_step(workload, hw, batch, reuse="statistical"):
    """decode_step with persistent caching (analytical namespace)."""
    from model import decode_step
    k = get(workload.model, workload.mode, "decode_step", batch,
            workload.context, hw.name, hw.org)
    if k is not None:
        return k
    r = decode_step(workload, hw, batch, reuse)
    v = {"latency_s": r.latency_s, "tps_per_user": r.tps_per_user,
         "aggregate_tps": r.aggregate_tps, "bytes_per_step": r.bytes_per_step,
         "dram_util": r.dram_util, "compute_util": r.compute_util,
         "simulation_method": "analytical"}
    put(workload.model, workload.mode, "decode_step", batch, workload.context,
        hw.name, hw.org, v)
    return v
