"""Kernel LUT: pulp_sim results -> kernel_cache (method=pulp_tile_sim) and a
full-model decode composer for the hetero tile on CUBE design points."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
import tile_sim
import kernel_cache


def compose_v4flash_decode(stacks=16, tiles_per_stack=4, batch=1,
                           context=131072, freq_ghz=1.0, mem_eff=0.8):
    """V4-Flash decode step on the hetero PULP die (method=pulp_tile_sim).

    Experts on NEUREKA (int4-as-fp4, ASSUMPTION P8); dense proj + attention
    + indexer on RedMulE fp16-rate with fp8 storage traffic; per-operator
    time = max(tile compute, stack memory). Memory efficiency from the
    calibrated pattern table is applied by caller via mem_eff.
    """
    from deepseek_v4_flash.dag import build_decode_dag
    from moe import unique_experts_statistical
    wl = build_decode_dag(context)
    tiles = stacks * tiles_per_stack
    bw = stacks * 0.256e12 * mem_eff
    t_total = 0.0
    detail = {}
    for op in wl.ops:
        # memory time for this op across the die
        if op.category == "routed_expert":
            esc = unique_experts_statistical(256, 6, batch) / 6
            step_bytes = op.weight_bytes * esc * op.count
            # NEUREKA path: fp4 traffic already in weight_bytes (fp4 in DAG)
            k = tile_sim.expert_gemv(4096, 2048, batch, "neureka", 4, freq_ghz)
            # unique experts spread across all tiles
            n_inst = 6 * batch * op.count / 6 * esc
            t_comp = k["seconds"] * (n_inst / tiles)
        else:
            step_bytes = (op.weight_bytes + (op.kv_read_bytes + op.kv_write_bytes
                          + op.index_read_bytes) * batch) * op.count
            flops = op.flops * batch * op.count
            # RedMulE fp16-rate across all tiles, swap-mapped: 48 MAC/c/tile
            t_comp = flops / 2 / (48e9 * freq_ghz * tiles * 0.95)
        t_mem = step_bytes / bw
        t_op = max(t_comp, t_mem)
        t_total += t_op
        c = detail.setdefault(op.category, [0.0, 0.0, 0.0])
        c[0] += t_comp; c[1] += t_mem; c[2] += t_op
    t_total += 60 * 3e-7        # per-layer sync (same constant as analytical)
    res = {"latency_s": t_total, "tps_per_user": 1 / t_total,
           "by_category": {k: {"t_comp": v[0], "t_mem": v[1], "t": v[2]}
                           for k, v in detail.items()},
           "method": "pulp_tile_sim"}
    kernel_cache.put("deepseek_v4_flash", "exact", "decode_step_pulp_hetero",
                     batch, context, f"pulp_hetero_{stacks}stacks", "pulp_hetero",
                     {"latency_s": t_total, "tps_per_user": 1 / t_total,
                      "simulation_method": "pulp_tile_sim"})
    return res
