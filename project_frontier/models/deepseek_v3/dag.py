"""DeepSeek-V3 (671B/37B) decode DAG — MLA absorbed form, per token, B=1.

Source of truth: official HF config (archived source_of_truth/deepseek_v3/).
Purpose (FabrikSim incorporation): V3 is the characterization vehicle for the
MLA arithmetic-intensity finding — the absorbed score path reads 576 B of
shared latent KV per token per layer while all 128 heads attend over it,
giving ~484 FLOP/byte vs 2-8 for weight streams. This DAG makes attention
FLOPs first-class (per-layer counts include the n_layers factor by
construction) and exposes an `attn_read_fraction` knob modeling
sparse-attention schemes that cap the fraction of context actually read.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workload_common import Op, Engine, AccessPattern, DecodeWorkload, BYTES

# --- exact hyperparameters (source: config.json) ---
DIM = 7168
N_LAYERS = 61
N_DENSE = 3               # first_k_dense_replace
N_MOE = 58
N_HEADS = 128
Q_LORA = 1536
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
N_EXPERTS = 256
TOPK = 8
N_SHARED = 1
MOE_INTER = 2048
DENSE_INTER = 18432
VOCAB = 129280

LATENT_B = (KV_LORA + QK_ROPE)      # 576 bytes/token/layer at fp8


def build_decode_dag(context: int, mode: str = "fp8",
                     attn_read_fraction: float = 1.0) -> DecodeWorkload:
    """attn_read_fraction < 1 models sparse attention (top-k/windowed) that
    reads only a fraction of the latent cache — clearly a labeled variant."""
    es = BYTES["fp8"] if mode == "fp8" else BYTES["bf16"]
    w = DecodeWorkload(model="deepseek_v3", mode=mode, context=context)
    ops = w.ops
    t_eff = max(1, int(context * attn_read_fraction))

    def gemv(name, cat, K, N, count, B=1, shared=True):
        return Op(name, cat, Engine.MATRIX, flops=2.0 * B * K * N,
                  weight_bytes=B * K * N * es, precision=mode,
                  shared_across_batch=shared, count=count)

    # ---- MLA projections (all layers) ----
    ops += [
        gemv("wq_a", "dense_proj", DIM, Q_LORA, N_LAYERS),
        gemv("wq_b", "dense_proj", Q_LORA, N_HEADS * (QK_NOPE + QK_ROPE), N_LAYERS),
        gemv("wkv_a", "dense_proj", DIM, KV_LORA + QK_ROPE, N_LAYERS),
        # absorbed W_UK: per-head q_nope[128] x [128,512] -> latent query
        gemv("uk_absorb", "dense_proj", QK_NOPE, KV_LORA, N_LAYERS, B=N_HEADS),
        # absorbed W_UV: per-head latent attn out [512] -> [128]
        gemv("uv_absorb", "dense_proj", KV_LORA, V_HEAD, N_LAYERS, B=N_HEADS),
        gemv("o_proj", "dense_proj", N_HEADS * V_HEAD, DIM, N_LAYERS),
    ]
    # ---- MLA attention core: ALL 128 heads over ONE shared 576B latent ----
    # QK over (kv_lora+rope) + SV over kv_lora; softmax ~3 ops/score.
    ops.append(Op("mla_attention", "full_attn", Engine.MATRIX,
                  flops=(2.0 * N_HEADS * (KV_LORA + QK_ROPE) * t_eff
                         + 2.0 * N_HEADS * KV_LORA * t_eff
                         + 3.0 * N_HEADS * t_eff),
                  kv_read_bytes=t_eff * LATENT_B * es,
                  kv_write_bytes=LATENT_B * es,
                  precision=mode, pattern=AccessPattern.SEQ_STREAM,
                  shared_across_batch=False, count=N_LAYERS,
                  notes=f"absorbed MLA: ~{2*N_HEADS*(KV_LORA+QK_ROPE+KV_LORA)//LATENT_B}"
                        " FLOP/byte on the latent stream"))
    # ---- FFN ----
    ops.append(gemv("dense_ffn", "dense_proj", DIM, 3 * DENSE_INTER, N_DENSE))
    ops.append(Op("router_gate", "router", Engine.MATRIX,
                  flops=2.0 * DIM * N_EXPERTS,
                  weight_bytes=N_EXPERTS * DIM * BYTES["bf16"], precision="bf16",
                  count=N_MOE))
    expert = 3 * DIM * MOE_INTER
    ops.append(Op("routed_experts", "routed_expert", Engine.MATRIX,
                  flops=2.0 * expert * TOPK + 8.0 * MOE_INTER * TOPK,
                  weight_bytes=expert * TOPK * es, precision=mode,
                  shared_across_batch=False, count=N_MOE,
                  notes="top-8 of 256; reuse via MoE model"))
    ops.append(Op("shared_expert", "shared_expert", Engine.MATRIX,
                  flops=2.0 * expert, weight_bytes=expert * es,
                  precision=mode, count=N_MOE))
    ops.append(Op("norms", "norm", Engine.VECTOR, flops=10.0 * DIM,
                  pattern=AccessPattern.ON_CHIP, count=N_LAYERS))
    ops.append(Op("lm_head", "head", Engine.MATRIX,
                  flops=2.0 * DIM * VOCAB, weight_bytes=DIM * VOCAB * es,
                  precision=mode, count=1))
    return w


def total_params():
    experts = N_MOE * N_EXPERTS * 3 * DIM * MOE_INTER
    shared = N_MOE * N_SHARED * 3 * DIM * MOE_INTER
    dense = N_DENSE * 3 * DIM * DENSE_INTER
    attn = N_LAYERS * (DIM * Q_LORA + Q_LORA * N_HEADS * (QK_NOPE + QK_ROPE)
                       + DIM * (KV_LORA + QK_ROPE)
                       + KV_LORA * N_HEADS * (QK_NOPE + V_HEAD)
                       + N_HEADS * V_HEAD * DIM)
    gate = N_MOE * N_EXPERTS * DIM
    embed_head = 2 * VOCAB * DIM
    return dict(experts=experts, shared=shared, dense_ffn=dense, attn=attn,
                gate=gate, embed_head=embed_head,
                total=experts + shared + dense + attn + gate + embed_head)


def kv_capacity_per_user(context: int, kv_bytes: float = 1.0) -> float:
    return N_LAYERS * context * LATENT_B * kv_bytes


MLA_FLOP_PER_BYTE = (2 * N_HEADS * (KV_LORA + QK_ROPE) + 2 * N_HEADS * KV_LORA) / LATENT_B
