"""GPT-OSS-120B decode DAG (per token, B=1). Source: official HF config
(openai/gpt-oss-120b, archived in source_of_truth/gpt_oss_120b/).

36 layers, hidden 2880, GQA 64Q/8KV x 64, alternating sliding_attention(128)
/ full_attention layers (18 each, from layer_types), 128 experts top-4
(gated FFN inter 2880, MXFP4 expert weights per quantization_config;
attention/router/embeddings unquantized -> modeled fp8 serving, labeled),
learned attention sinks, vocab 201088, max ctx 131072.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workload_common import Op, Engine, AccessPattern, DecodeWorkload, BYTES

DIM = 2880
N_LAYERS = 36
N_SWA = 18
N_FULL = 18
N_HEADS = 64
N_KV = 8
HEAD_DIM = 64
WINDOW = 128
N_EXPERTS = 128
TOPK = 4
INTER = 2880
VOCAB = 201088

FP4, FP8, BF16 = BYTES["fp4"], BYTES["fp8"], BYTES["bf16"]
KV_TOK = N_KV * HEAD_DIM * 2          # elements per token per layer


def build_decode_dag(context: int, mode: str = "exact_fp8kv") -> DecodeWorkload:
    kp = FP8 if mode.endswith("fp8kv") else BF16
    w = DecodeWorkload(model="gpt_oss_120b", mode=mode, context=context)
    ops = w.ops

    def gemv(name, cat, K, N, count, prec=FP8, pl="fp8", shared=True):
        return Op(name, cat, Engine.MATRIX, flops=2.0 * K * N,
                  weight_bytes=K * N * prec, precision=pl,
                  shared_across_batch=shared, count=count)

    qkv_n = (N_HEADS + 2 * N_KV) * HEAD_DIM
    ops += [
        gemv("qkv_proj", "dense_proj", DIM, qkv_n, N_LAYERS),
        gemv("o_proj", "dense_proj", N_HEADS * HEAD_DIM, DIM, N_LAYERS),
        Op("norms_rope_sinks", "norm", Engine.VECTOR, flops=8.0 * DIM + 2 * N_HEADS,
           pattern=AccessPattern.ON_CHIP, count=N_LAYERS),
        Op("kv_write", "local_attn", Engine.MEMORY, kv_write_bytes=KV_TOK * kp,
           pattern=AccessPattern.RANDOM_GATHER, shared_across_batch=False,
           count=N_LAYERS),
    ]
    span_w = min(context, WINDOW)
    ops.append(Op("swa_attn", "local_attn", Engine.MATRIX,
                  flops=2.0 * N_HEADS * HEAD_DIM * span_w * 2 + 3 * N_HEADS * span_w,
                  kv_read_bytes=span_w * KV_TOK * kp,
                  pattern=AccessPattern.BLOCK_GATHER, shared_across_batch=False,
                  count=N_SWA))
    ops.append(Op("full_attn", "full_attn", Engine.MATRIX,
                  flops=2.0 * N_HEADS * HEAD_DIM * context * 2 + 3 * N_HEADS * context,
                  kv_read_bytes=context * KV_TOK * kp,
                  pattern=AccessPattern.SEQ_STREAM, shared_across_batch=False,
                  count=N_FULL))
    ops.append(Op("router", "router", Engine.MATRIX, flops=2.0 * DIM * N_EXPERTS,
                  weight_bytes=N_EXPERTS * DIM * BF16, precision="bf16",
                  count=N_LAYERS))
    expert = 3 * DIM * INTER
    ops.append(Op("routed_experts", "routed_expert", Engine.MATRIX,
                  flops=2.0 * expert * TOPK + 8.0 * INTER * TOPK,
                  weight_bytes=expert * TOPK * FP4, precision="fp4",
                  shared_across_batch=False, count=N_LAYERS,
                  notes="MXFP4 per official quantization_config; top-4 of 128"))
    ops.append(Op("lm_head", "head", Engine.MATRIX, flops=2.0 * DIM * VOCAB,
                  weight_bytes=DIM * VOCAB * FP8, precision="fp8", count=1))
    return w


def total_params():
    experts = N_LAYERS * N_EXPERTS * 3 * DIM * INTER
    attn = N_LAYERS * (DIM * (N_HEADS + 2 * N_KV) * HEAD_DIM + N_HEADS * HEAD_DIM * DIM)
    gate = N_LAYERS * N_EXPERTS * DIM
    embed_head = 2 * VOCAB * DIM
    return dict(experts=experts, attn=attn, gate=gate, embed_head=embed_head,
                total=experts + attn + gate + embed_head)


def weight_capacity_gb():
    p = total_params()
    return (p["experts"] * BYTES["fp4"] + (p["attn"] + p["gate"]) * BYTES["fp8"]
            + p["embed_head"] * BYTES["fp8"]) / 1e9


def kv_capacity_per_user(context: int, kv_bytes: float = 1.0) -> float:
    return (N_FULL * context + N_SWA * min(context, WINDOW)) * KV_TOK * kv_bytes
