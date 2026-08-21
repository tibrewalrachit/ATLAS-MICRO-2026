"""MiniMax M3 decode operator DAG (per token, per user, B=1 basis).

Dimensions from the official HF config (revision f0e1c1e), archived under
models/source_of_truth/minimax_m3/. MSA mechanism semantics (max-pool block
scoring, index heads, local-block inclusion, KV-outer-gather-Q) from the MSA
paper (arXiv:2606.13392); every M3 number comes from the config itself.

Layer taxonomy (moe_layer_freq / sparse_attention_freq = [0,0,0]+[1]*57):
  - 3  dense layers: dense FFN (inter 12288) + FULL softmax attention
  - 57 MoE+MSA layers: 128 experts top-4 + 1 shared (inter 3072),
       MSA: top-16 blocks x 128 tokens (+ local block), 4 index heads x 128
Text decode only; vision tower excluded. MTP modules (7) off in baseline.

Precision modes (weights): checkpoint is bf16; serving precision is NOT
published. mode="exact_bf16" uses bf16 weights+KV; mode="exact_fp8" uses fp8
weights+KV (labeled assumption, standard for high-throughput serving).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workload_common import Op, Engine, AccessPattern, DecodeWorkload, BYTES

# --- exact hyperparameters (source: config.json text_config @f0e1c1e) ---
DIM = 6144
N_LAYERS = 60
N_DENSE = 3            # full-attention + dense-FFN layers
N_MOE = 57
N_HEADS = 64
N_KV_HEADS = 4
HEAD_DIM = 128
DENSE_INTER = 12288
EXPERT_INTER = 3072
SHARED_INTER = 3072
N_EXPERTS = 128
TOPK = 4
VOCAB = 200064
BLOCK = 128            # sparse_block_size
TOPK_BLOCKS = 16       # sparse_topk_blocks
IDX_HEADS = 4          # sparse_num_index_heads
IDX_DIM = 128          # sparse_index_dim
LOCAL_BLOCKS = 1       # sparse_local_block

KV_PER_TOKEN_EL = N_KV_HEADS * HEAD_DIM * 2      # K+V elements per token per layer


def build_decode_dag(context: int, mode: str = "exact_fp8") -> DecodeWorkload:
    assert mode in ("exact_fp8", "exact_bf16")
    wp = BYTES["fp8"] if mode == "exact_fp8" else BYTES["bf16"]   # weight precision
    kp = wp                                                        # KV precision follows mode
    w = DecodeWorkload(model="minimax_m3", mode=mode, context=context)
    t = context
    ops = w.ops

    def gemv(name, cat, K, N, count, shared=True, pattern=AccessPattern.SEQ_STREAM):
        return Op(name=name, category=cat, engine=Engine.MATRIX, flops=2.0 * K * N,
                  weight_bytes=K * N * wp, precision=w.mode.split("_")[1],
                  pattern=pattern, shared_across_batch=shared, count=count)

    # ---------- attention projections (all 60 layers) ----------
    qkv_out = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
    ops += [
        gemv("qkv_proj", "dense_proj", DIM, qkv_out, N_LAYERS),
        gemv("o_proj", "dense_proj", N_HEADS * HEAD_DIM, DIM, N_LAYERS),
        Op("qk_norm_rope", "norm", Engine.VECTOR,
           flops=4.0 * N_HEADS * HEAD_DIM + 2 * DIM * 4,
           pattern=AccessPattern.ON_CHIP, count=N_LAYERS),
    ]
    # KV write each layer
    ops.append(Op("kv_write", "full_attn", Engine.MEMORY,
                  kv_write_bytes=KV_PER_TOKEN_EL * kp,
                  pattern=AccessPattern.RANDOM_GATHER, shared_across_batch=False,
                  count=N_LAYERS))

    # ---------- 3 full-attention layers ----------
    # GQA: 64 q heads share 4 KV heads; KV read once per group set.
    ops.append(Op("full_attn", "full_attn", Engine.MATRIX,
                  flops=2.0 * N_HEADS * HEAD_DIM * t * 2 + 3.0 * N_HEADS * t,
                  kv_read_bytes=t * KV_PER_TOKEN_EL * kp,
                  precision=w.mode.split("_")[1], pattern=AccessPattern.SEQ_STREAM,
                  shared_across_batch=False, count=N_DENSE,
                  notes="full softmax attention over entire context"))

    # ---------- 57 MSA layers ----------
    n_blocks = max(1, t // BLOCK)
    sel_blocks = min(TOPK_BLOCKS + LOCAL_BLOCKS, n_blocks)
    sel_tokens = sel_blocks * BLOCK if t >= BLOCK else t
    # index q projection (1 idx q head per GQA group; shared idx key from K pooling)
    ops.append(gemv("msa_idx_q", "msa_index", DIM, IDX_HEADS * IDX_DIM, N_MOE))
    # block-score scan: q[4,128] . block_keys[n_blocks, 128] per idx head
    ops.append(Op("msa_idx_scan", "msa_index", Engine.INDEX,
                  flops=2.0 * IDX_HEADS * IDX_DIM * n_blocks + IDX_HEADS * n_blocks,
                  index_read_bytes=n_blocks * IDX_DIM * kp,
                  precision=w.mode.split("_")[1], pattern=AccessPattern.SCAN,
                  shared_across_batch=False, count=N_MOE,
                  notes="max-pool block keys scan + top-16 select"))
    # index-key maintenance: pooled block key update (amortized per block fill)
    ops.append(Op("msa_idx_write", "msa_index", Engine.MEMORY,
                  kv_write_bytes=IDX_DIM * kp / BLOCK,
                  pattern=AccessPattern.SEQ_STREAM, shared_across_batch=False, count=N_MOE))
    # sparse attention over selected blocks (contiguous 128-token blocks)
    ops.append(Op("msa_attn", "msa_attn", Engine.MATRIX,
                  flops=2.0 * N_HEADS * HEAD_DIM * sel_tokens * 2 + 3.0 * N_HEADS * sel_tokens,
                  kv_read_bytes=sel_tokens * KV_PER_TOKEN_EL * kp,
                  precision=w.mode.split("_")[1], pattern=AccessPattern.BLOCK_GATHER,
                  shared_across_batch=False, count=N_MOE,
                  notes="top-16+local 128-token blocks; KV-outer-gather-Q contiguous"))

    # ---------- FFN ----------
    # 3 dense layers: gated FFN inter 12288
    ops.append(gemv("dense_ffn", "dense_proj", DIM, 3 * DENSE_INTER, N_DENSE,
                    pattern=AccessPattern.SEQ_STREAM))
    # 57 MoE layers
    ops.append(Op("router_gate", "router", Engine.MATRIX,
                  flops=2.0 * DIM * N_EXPERTS + 6 * N_EXPERTS,
                  weight_bytes=N_EXPERTS * DIM * BYTES["bf16"], precision="bf16",
                  pattern=AccessPattern.SEQ_STREAM, count=N_MOE))
    expert_params = 3 * DIM * EXPERT_INTER
    ops.append(Op("routed_experts", "routed_expert", Engine.MATRIX,
                  flops=2.0 * expert_params * TOPK + 8.0 * EXPERT_INTER * TOPK,
                  weight_bytes=expert_params * TOPK * wp,
                  precision=w.mode.split("_")[1], pattern=AccessPattern.SEQ_STREAM,
                  shared_across_batch=False, count=N_MOE,
                  notes="top-4 of 128; batch reuse via MoE reuse model"))
    ops.append(Op("shared_expert", "shared_expert", Engine.MATRIX,
                  flops=2.0 * 3 * DIM * SHARED_INTER + 8.0 * SHARED_INTER,
                  weight_bytes=3 * DIM * SHARED_INTER * wp,
                  precision=w.mode.split("_")[1], pattern=AccessPattern.SEQ_STREAM,
                  count=N_MOE))
    # norms/residual
    ops.append(Op("norms_resid", "norm", Engine.VECTOR, flops=12.0 * DIM,
                  pattern=AccessPattern.ON_CHIP, count=N_LAYERS))

    # ---------- LM head ----------
    ops.append(Op("lm_head", "head", Engine.MATRIX,
                  flops=2.0 * DIM * VOCAB, weight_bytes=DIM * VOCAB * BYTES["bf16"],
                  precision="bf16", pattern=AccessPattern.SEQ_STREAM, count=1))
    ops.append(Op("embed", "head", Engine.MEMORY, weight_bytes=DIM * BYTES["bf16"],
                  precision="bf16", pattern=AccessPattern.RANDOM_GATHER,
                  shared_across_batch=False, count=1))
    return w


def total_params():
    experts = N_MOE * N_EXPERTS * 3 * DIM * EXPERT_INTER
    shared = N_MOE * 3 * DIM * SHARED_INTER
    dense_ffn = N_DENSE * 3 * DIM * DENSE_INTER
    attn = N_LAYERS * (DIM * (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
                       + N_HEADS * HEAD_DIM * DIM)
    idx = N_MOE * DIM * IDX_HEADS * IDX_DIM
    gate = N_MOE * N_EXPERTS * DIM
    embed_head = 2 * VOCAB * DIM
    return dict(experts=experts, shared=shared, dense_ffn=dense_ffn, attn=attn,
                indexer=idx, gate=gate, embed_head=embed_head,
                total=experts + shared + dense_ffn + attn + idx + gate + embed_head)


def kv_capacity_per_user(context: int, kv_bytes_per_el: float = 1.0) -> float:
    kv = N_LAYERS * context * KV_PER_TOKEN_EL * kv_bytes_per_el
    idx = N_MOE * (context // BLOCK) * IDX_DIM * kv_bytes_per_el
    return kv + idx
