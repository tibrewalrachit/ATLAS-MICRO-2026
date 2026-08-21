"""DeepSeek V4-Flash decode operator DAG (per token, per user, B=1 basis).

Every dimension is taken from the official HF repo (revision 60d8d70), archived
under models/source_of_truth/deepseek_v4_flash/:
  - config.json / inference/config.json  (all hyperparameters)
  - inference/model.py                   (operator semantics: Attention,
                                          Compressor, Indexer, MoE, Block/HC)

Layer taxonomy from compress_ratios = [0,0] + [4,128]*20 + [4] (+[0] for MTP):
  - 2  SWA layers  (ratio 0):   sliding-window-128 MQA only
  - 21 CSA layers  (ratio 4):   window + Indexer top-512 over 4:1-compressed KV
  - 20 HCA layers  (ratio 128): window + ALL t/128-compressed KV
All 43 layers are MoE (256 routed FP4, top-6, 1 shared FP8, inter 2048).
MTP (1 layer) is excluded: baseline decode has speculative decoding off.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workload_common import Op, Engine, AccessPattern, DecodeWorkload, BYTES

# --- exact hyperparameters (source: config.json @60d8d70) ---
DIM = 4096
N_LAYERS = 43
N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA = 1024
O_GROUPS = 8
O_LORA = 1024
WINDOW = 128
N_EXPERTS = 256
TOPK = 6
N_SHARED = 1
MOE_INTER = 2048
IDX_HEADS = 64
IDX_DIM = 128
IDX_TOPK = 512
HC_MULT = 4
VOCAB = 129280
N_SWA, N_CSA, N_HCA = 2, 21, 20   # from compress_ratios

FP4, FP8, BF16, FP32 = BYTES["fp4"], BYTES["fp8"], BYTES["bf16"], BYTES["fp32"]

# KV entry: head_dim 512 of which rope 64 stays bf16, non-rope 448 is fp8
KV_ENTRY_B = (HEAD_DIM - ROPE_DIM) * FP8 + ROPE_DIM * BF16          # 576 B
IDX_KV_ENTRY_B = IDX_DIM * FP4                                       # ~68 B (fp4)

def gemv(name, cat, K, N, wbytes_per_param, shared=True, notes="", engine=Engine.MATRIX,
         pattern=AccessPattern.SEQ_STREAM, count=1, precision="fp8"):
    return Op(name=name, category=cat, engine=engine, flops=2.0 * K * N,
              weight_bytes=K * N * wbytes_per_param, precision=precision,
              pattern=pattern, shared_across_batch=shared, count=count, notes=notes)


def attn_core(name, cat, span_tokens, kv_entry_bytes, pattern, count=1, notes=""):
    """MQA attention: 64 q heads against a single shared KV of head_dim 512.
    FLOPs: QK + SV = 2 * heads * head_dim * span * 2. KV read once (MQA)."""
    return Op(name=name, category=cat, engine=Engine.MATRIX,
              flops=2.0 * N_HEADS * HEAD_DIM * span_tokens * 2 + 3.0 * N_HEADS * span_tokens,
              kv_read_bytes=span_tokens * kv_entry_bytes,
              precision="fp8", pattern=pattern, shared_across_batch=False,
              count=count, notes=notes)


def build_decode_dag(context: int, mode: str = "exact") -> DecodeWorkload:
    w = DecodeWorkload(model="deepseek_v4_flash", mode=mode, context=context)
    t = context
    ops = w.ops

    # ---------- per-layer attention-side dense projections (all 43 layers) ----------
    ops += [
        gemv("wq_a", "dense_proj", DIM, Q_LORA, FP8, count=N_LAYERS),
        gemv("wq_b", "dense_proj", Q_LORA, N_HEADS * HEAD_DIM, FP8, count=N_LAYERS),
        gemv("wkv", "dense_proj", DIM, HEAD_DIM, FP8, count=N_LAYERS),
        # grouped low-rank O: wo_a (fp8 per code comment), wo_b
        gemv("wo_a", "dense_proj", N_HEADS * HEAD_DIM // O_GROUPS, O_GROUPS * O_LORA, FP8, count=N_LAYERS),
        gemv("wo_b", "dense_proj", O_GROUPS * O_LORA, DIM, FP8, count=N_LAYERS),
    ]
    # norms + rope + hc residual mixing (vector engine, activations on-chip)
    hc_params = 2 * ((2 + HC_MULT) * HC_MULT * HC_MULT * DIM)   # attn+ffn hc_fn matrices
    ops += [
        Op("hc_mix", "residual_hc", Engine.VECTOR,
           flops=2.0 * hc_params + 6 * HC_MULT * DIM + 20 * 36,   # F.linear mixes + sinkhorn(20 it, 6x6)
           weight_bytes=hc_params * FP32, precision="fp32",
           pattern=AccessPattern.SEQ_STREAM, count=N_LAYERS,
           notes="hyper-connection pre/post mixing, hc_mult=4"),
        Op("norms_rope", "norm", Engine.VECTOR,
           flops=10.0 * DIM + 2 * N_HEADS * HEAD_DIM + 4 * ROPE_DIM * N_HEADS,
           pattern=AccessPattern.ON_CHIP, count=N_LAYERS),
    ]

    # ---------- KV write (every layer writes one entry to window ring) ----------
    ops.append(Op("kv_write", "local_attn", Engine.MEMORY,
                  kv_write_bytes=KV_ENTRY_B, pattern=AccessPattern.RANDOM_GATHER,
                  shared_across_batch=False, count=N_LAYERS))

    # ---------- SWA layers (2): window attention only ----------
    span_w = min(t, WINDOW)
    ops.append(attn_core("swa_attn", "local_attn", span_w, KV_ENTRY_B,
                         AccessPattern.BLOCK_GATHER, count=N_SWA))

    # ---------- CSA layers (21) ----------
    n_comp4 = t // 4
    span_csa = span_w + min(IDX_TOPK, n_comp4)
    ops.append(attn_core("csa_attn", "csa_attn", span_csa, KV_ENTRY_B,
                         AccessPattern.BLOCK_GATHER, count=N_CSA,
                         notes="window + top-512 of 4:1 compressed KV (rows scattered)"))
    # main compressor: wkv+wgate (bf16 checkpoint, fp32 compute), coff=2 (overlap)
    ops.append(gemv("csa_compressor", "csa_attn", DIM, 2 * 2 * HEAD_DIM, BF16,
                    count=N_CSA, precision="bf16",
                    notes="Compressor wkv+wgate, overlap; runs every token"))
    # compressed-KV write amortized: 1 entry per 4 tokens
    ops.append(Op("csa_ckv_write", "csa_attn", Engine.MEMORY,
                  kv_write_bytes=KV_ENTRY_B / 4, pattern=AccessPattern.SEQ_STREAM,
                  shared_across_batch=False, count=N_CSA))
    # Indexer: wq_b + weights_proj + its own compressor (head_dim 128)
    ops.append(gemv("idx_q_proj", "indexer", Q_LORA, IDX_HEADS * IDX_DIM, FP8, count=N_CSA))
    ops.append(gemv("idx_weights_proj", "indexer", DIM, IDX_HEADS, BF16, count=N_CSA, precision="bf16"))
    ops.append(gemv("idx_compressor", "indexer", DIM, 2 * 2 * IDX_DIM, BF16,
                    count=N_CSA, precision="bf16"))
    # Index score scan: q[64,128] x idx_kv[t/4,128] fp4 + relu + weighted sum + topk
    ops.append(Op("idx_scan", "indexer", Engine.INDEX,
                  flops=2.0 * IDX_HEADS * IDX_DIM * n_comp4 + 2.0 * IDX_HEADS * n_comp4,
                  index_read_bytes=n_comp4 * IDX_KV_ENTRY_B,
                  precision="fp4", pattern=AccessPattern.SCAN,
                  shared_across_batch=False, count=N_CSA,
                  notes="scans ALL t/4 compressed index entries; top-512 select"))
    ops.append(Op("idx_ckv_write", "indexer", Engine.MEMORY,
                  kv_write_bytes=IDX_KV_ENTRY_B / 4, pattern=AccessPattern.SEQ_STREAM,
                  shared_across_batch=False, count=N_CSA))

    # ---------- HCA layers (20) ----------
    n_comp128 = t // 128
    span_hca = span_w + n_comp128
    ops.append(attn_core("hca_attn", "hca_attn", span_hca, KV_ENTRY_B,
                         AccessPattern.SEQ_STREAM, count=N_HCA,
                         notes="window + ALL t/128 compressed KV (contiguous scan)"))
    ops.append(gemv("hca_compressor", "hca_attn", DIM, 2 * HEAD_DIM, BF16,
                    count=N_HCA, precision="bf16", notes="no overlap at ratio 128"))
    ops.append(Op("hca_ckv_write", "hca_attn", Engine.MEMORY,
                  kv_write_bytes=KV_ENTRY_B / 128, pattern=AccessPattern.SEQ_STREAM,
                  shared_across_batch=False, count=N_HCA))

    # ---------- MoE (all 43 layers) ----------
    ops.append(Op("router_gate", "router", Engine.MATRIX,
                  flops=2.0 * DIM * N_EXPERTS + 5 * N_EXPERTS,
                  weight_bytes=N_EXPERTS * DIM * BF16, precision="bf16",
                  pattern=AccessPattern.SEQ_STREAM, count=N_LAYERS))
    expert_params = 3 * DIM * MOE_INTER
    ops.append(Op("routed_experts", "routed_expert", Engine.MATRIX,
                  flops=2.0 * expert_params * TOPK + 8.0 * MOE_INTER * TOPK,
                  weight_bytes=expert_params * TOPK * FP4, precision="fp4",
                  pattern=AccessPattern.SEQ_STREAM, shared_across_batch=False,
                  count=N_LAYERS,
                  notes="top-6 of 256, FP4; batch reuse via MoE reuse model"))
    ops.append(Op("shared_expert", "shared_expert", Engine.MATRIX,
                  flops=2.0 * expert_params + 8.0 * MOE_INTER,
                  weight_bytes=expert_params * FP8, precision="fp8",
                  pattern=AccessPattern.SEQ_STREAM, count=N_LAYERS))

    # ---------- LM head (once) ----------
    ops.append(Op("lm_head", "head", Engine.MATRIX,
                  flops=2.0 * DIM * VOCAB, weight_bytes=DIM * VOCAB * BF16,
                  precision="bf16", pattern=AccessPattern.SEQ_STREAM, count=1))
    ops.append(Op("embed", "head", Engine.MEMORY,
                  weight_bytes=DIM * BF16, precision="bf16",
                  pattern=AccessPattern.RANDOM_GATHER, shared_across_batch=False, count=1))
    return w


# --- static reference quantities (for validation & the report) ---
def total_params():
    routed = N_LAYERS * N_EXPERTS * 3 * DIM * MOE_INTER
    shared = N_LAYERS * N_SHARED * 3 * DIM * MOE_INTER
    attn = N_LAYERS * (DIM * Q_LORA + Q_LORA * N_HEADS * HEAD_DIM + DIM * HEAD_DIM
                       + (N_HEADS * HEAD_DIM // O_GROUPS) * O_GROUPS * O_LORA
                       + O_GROUPS * O_LORA * DIM)
    idx = N_CSA * (Q_LORA * IDX_HEADS * IDX_DIM + DIM * IDX_HEADS + DIM * 4 * IDX_DIM)
    comp = N_CSA * DIM * 4 * HEAD_DIM + N_HCA * DIM * 2 * HEAD_DIM
    gate = N_LAYERS * N_EXPERTS * DIM
    hc = N_LAYERS * 2 * (2 + HC_MULT) * HC_MULT * HC_MULT * DIM
    embed_head = 2 * VOCAB * DIM
    return dict(routed=routed, shared=shared, attn=attn, indexer=idx,
                compressors=comp, gate=gate, hc=hc, embed_head=embed_head,
                total=routed + shared + attn + idx + comp + gate + hc + embed_head)


def kv_capacity_per_user(context: int) -> float:
    """Bytes of KV + index state per user at a given context."""
    win = N_LAYERS * WINDOW * KV_ENTRY_B
    csa = N_CSA * (context // 4) * (KV_ENTRY_B + IDX_KV_ENTRY_B / 4 * 4)  # ckv + idx ckv
    csa = N_CSA * ((context // 4) * KV_ENTRY_B + (context // 4) * IDX_KV_ENTRY_B)
    hca = N_HCA * (context // 128) * KV_ENTRY_B
    return win + csa + hca
