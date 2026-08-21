"""Closed-form attention/KV byte formulas — independent cross-check of the DAG
builders (validation gate, Part XXXVI step 5). Derived directly from the
official model code semantics, written separately from dag.py on purpose."""

def v4_kv_read_bytes_per_token(ctx: int) -> float:
    KV = 448 * 1.0 + 64 * 2.0        # fp8 non-rope + bf16 rope
    win = min(ctx, 128)
    swa = 2 * win * KV
    csa = 21 * (win + min(512, ctx // 4)) * KV
    hca = 20 * (win + ctx // 128) * KV
    return swa + csa + hca

def v4_index_bytes_per_token(ctx: int) -> float:
    return 21 * (ctx // 4) * 128 * 0.5 * (1 + 1 / 16)   # fp4 + scales

def m3_kv_read_bytes_per_token(ctx: int, kv_bytes: float = 1.0) -> float:
    per_tok = 4 * 128 * 2 * kv_bytes
    full = 3 * ctx * per_tok
    sel = min(17 * 128, ctx)
    msa = 57 * sel * per_tok
    return full + msa

def m3_index_bytes_per_token(ctx: int, kv_bytes: float = 1.0) -> float:
    return 57 * (ctx // 128) * 128 * kv_bytes
