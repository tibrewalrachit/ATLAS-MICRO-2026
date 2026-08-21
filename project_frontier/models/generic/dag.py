"""Generic GQA/MoE decode DAG for ATLAS-supported models (opt, mixtral, qwen3).

Purpose (Part XIX): the analytical model must reproduce ATLAS cycle-level
results on models ATLAS supports natively, before we trust it on V4-Flash/M3.
Mirrors frontend/model_parser.get_layer_operator_list semantics, including
megatron TP sharding, so per-chip traffic matches what ATLAS simulates.
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workload_common import Op, Engine, AccessPattern, DecodeWorkload


def build_decode_dag(config_path: str, context: int, element_size: float = 2,
                     tp: int = 8, ep: int = 8, name: str = "",
                     expert_element_size: float = None) -> DecodeWorkload:
    cfg = json.load(open(config_path))
    hid = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_head)
    head_dim = cfg.get("head_dim", hid // n_head)
    inter = cfg.get("intermediate_size", 4 * hid)
    vocab = cfg["vocab_size"]
    n_exp = cfg.get("num_experts", cfg.get("num_local_experts", 1))
    top_k = cfg.get("num_experts_per_tok", 1) if n_exp > 1 else 1
    moe_inter = cfg.get("moe_intermediate_size", inter)
    gated = cfg.get("hidden_act", "") == "silu"
    es = float(element_size)

    w = DecodeWorkload(model=name or os.path.basename(config_path), mode="generic",
                       context=context)
    w.moe_shape = (n_exp, top_k)
    ops = w.ops
    # per-TP-shard shapes (megatron)
    import math
    qkv_n = head_dim * (math.ceil(n_head / tp) + 2 * math.ceil(n_kv / tp))
    ops.append(Op("qkv_proj", "dense_proj", Engine.MATRIX,
                  flops=2.0 * hid * qkv_n, weight_bytes=hid * qkv_n * es,
                  count=n_layers))
    kv_heads_shard = math.ceil(n_kv / tp)
    span = context
    ops.append(Op("attention", "full_attn", Engine.MATRIX,
                  flops=(2.0 * kv_heads_shard * (n_head // n_kv) * head_dim * span * 2
                         + 3.0 * kv_heads_shard * (n_head // n_kv) * span),
                  kv_read_bytes=kv_heads_shard * span * head_dim * 2 * es,
                  kv_write_bytes=kv_heads_shard * head_dim * 2 * es,
                  pattern=AccessPattern.SEQ_STREAM, shared_across_batch=False,
                  count=n_layers))
    o_k = head_dim * math.ceil(n_head / tp)
    ops.append(Op("o_proj", "dense_proj", Engine.MATRIX,
                  flops=2.0 * o_k * hid, weight_bytes=o_k * hid * es, count=n_layers))
    ffn_inter = moe_inter if n_exp > 1 else inter
    gate_f = 2 if gated else 1
    if n_exp > 1:
        # EP shards experts across ep chips: per-chip resident experts =
        # n_exp/ep; expected traffic on this chip = (unique-expert stream)/ep
        # under uniform expert placement (validated vs ATLAS mixtral case).
        ees = expert_element_size if expert_element_size is not None else es
        ops.append(Op("routed_experts", "routed_expert", Engine.MATRIX,
                      flops=2.0 * (gate_f + 1) * hid * ffn_inter * top_k / ep,
                      weight_bytes=(gate_f + 1) * hid * ffn_inter * top_k * ees / ep,
                      shared_across_batch=False, count=n_layers))
        ops.append(Op("router", "router", Engine.MATRIX,
                      flops=2.0 * hid * n_exp, weight_bytes=hid * n_exp * es,
                      count=n_layers))
    else:
        ops.append(Op("ffn_up", "dense_proj", Engine.MATRIX,
                      flops=2.0 * hid * (ffn_inter * gate_f) / tp,
                      weight_bytes=hid * ffn_inter * gate_f / tp * es, count=n_layers))
        ops.append(Op("ffn_down", "dense_proj", Engine.MATRIX,
                      flops=2.0 * ffn_inter / tp * hid,
                      weight_bytes=ffn_inter / tp * hid * es, count=n_layers))
    ops.append(Op("norms", "norm", Engine.VECTOR, flops=8.0 * hid,
                  pattern=AccessPattern.ON_CHIP, count=n_layers))
    ops.append(Op("lm_head", "head", Engine.MATRIX,
                  flops=2.0 * hid * vocab / tp, weight_bytes=hid * vocab / tp * es,
                  count=1))
    return w
