#!/usr/bin/env python3
"""What a fixed-size recurrent state does to an ATLAS chip's memory budget.

A conventional KV cache grows with context; a KDA state does not.  For a
design whose whole premise is bonded-DRAM bandwidth and whose capacity is set
by the DRAM dies it can bond, that is a capacity question before it is a
bandwidth one, so it is worth working out against the actual numbers rather
than asserted.

Model geometry from configs/models/qwen3_235b_a22b.json; DRAM organisation
from configs/architecture/dram/cloud/cloud_1TBps.yaml and the HBDRAM preset in
patches/ramulator2.patch; parallelism from
configs/architecture/system/test_cloud_system.yaml.
"""

GB = 1024.0 ** 3
MB = 1024.0 ** 2

# --- Qwen3-235B-A22B -------------------------------------------------------
LAYERS      = 94
KV_HEADS    = 4          # grouped-query attention
HEAD_DIM    = 128
TOTAL_PARAM = 235e9
ACTIVE_PARAM = 22e9      # A22B: active parameters per token

# --- ATLAS cloud chip ------------------------------------------------------
CHANNELS    = 16
DENSITY_GB  = 4          # HBDRAM_4Gb_1024pin_512col: 4 Gbit per channel
CHIP_BYTES  = CHANNELS * DENSITY_GB * 1e9 / 8          # gigabit -> byte
TP, EP      = 8, 8       # test_cloud_system.yaml
CHIPS       = TP * EP

# --- KDA, per the blog: fixed d x d state, 3:1 KDA to full-attention layers -
KDA_RATIO   = 0.75
STATE_BYTES = 4          # binary32 state, as implemented in rtl/attn


def kv_bytes_per_token(layers, kv_elem_bytes):
    """Two tensors (K and V), per KV head, per layer."""
    return 2 * layers * KV_HEADS * HEAD_DIM * kv_elem_bytes


def main():
    print("=" * 72)
    print("Memory footprint: growing KV cache vs a fixed KDA state")
    print("=" * 72)
    print()
    print("Qwen3-235B-A22B: %d layers, %d KV heads, head_dim %d"
          % (LAYERS, KV_HEADS, HEAD_DIM))
    print("ATLAS cloud chip: %d HBDRAM channels x %d Gbit = %.1f GB per chip"
          % (CHANNELS, DENSITY_GB, CHIP_BYTES / GB))
    print("System: tp=%d, ep=%d -> %d chips, %.1f GB aggregate"
          % (TP, EP, CHIPS, CHIPS * CHIP_BYTES / GB))
    print()

    # Weights, at the precisions this RTL implements.
    print("Weights")
    print("-" * 72)
    for name, b in [("MXFP4 (0.5 B/param + E8M0 per 32)", 0.5 + 1.0 / 32),
                    ("FP8   (1 B/param)", 1.0)]:
        tot = TOTAL_PARAM * b
        print("  %-36s %8.1f GB total, %6.2f GB per chip"
              % (name, tot / GB, tot / CHIPS / GB))
    print()

    n_kda = int(LAYERS * KDA_RATIO)
    n_full = LAYERS - n_kda

    print("KV cache vs KDA state, at FP8 KV")
    print("-" * 72)
    print("  %-10s %14s %14s %10s" % ("context", "all-attention", "KDA 3:1", "ratio"))

    kda_state = n_kda * KV_HEADS * HEAD_DIM * HEAD_DIM * STATE_BYTES

    for ctx in (4096, 32768, 131072, 524288):
        full = kv_bytes_per_token(LAYERS, 1.0) * ctx
        mixed = kv_bytes_per_token(n_full, 1.0) * ctx + kda_state
        print("  %-10s %11.2f GB %11.2f GB %9.1fx"
              % ("%dK" % (ctx // 1024), full / GB, mixed / GB, full / mixed))
    print()
    print("  The KDA half of that is %.1f MB and does not move with context;"
          % (kda_state / MB))
    print("  %d of %d layers keep a conventional cache." % (n_full, LAYERS))
    print()

    # The capacity question.
    #
    # Attention is sharded by tensor parallelism, so a chip holds 1/TP of the
    # KV.  Expert parallelism shards the MLP, not attention, so each expert
    # group carries its own copy -- dividing by all 64 chips would be wrong and
    # would flatter the result by 8x.
    print("Does it fit? (per chip: weights at MXFP4 + this chip's KV share)")
    print("-" * 72)
    w_chip = TOTAL_PARAM * (0.5 + 1.0 / 32) / CHIPS
    head = CHIP_BYTES - w_chip
    print("  weights per chip                      %6.2f GB" % (w_chip / GB))
    print("  HBDRAM per chip                       %6.2f GB" % (CHIP_BYTES / GB))
    print("  left for KV                           %6.2f GB" % (head / GB))
    print("  KV is sharded across tp=%d, replicated across ep=%d" % (TP, EP))
    print()
    print("  %-6s %-8s %14s %8s %14s %8s"
          % ("batch", "context", "all-attention", "", "KDA 3:1", ""))
    for batch in (1, 8, 32):
        for ctx in (32768, 131072):
            full = kv_bytes_per_token(LAYERS, 1.0) * ctx * batch / TP
            mixed = (kv_bytes_per_token(n_full, 1.0) * ctx * batch
                     + kda_state * batch) / TP
            print("  %-6d %-8s %11.2f GB %8s %11.2f GB %8s"
                  % (batch, "%dK" % (ctx // 1024),
                     full / GB, "fits" if full < head else "OVER",
                     mixed / GB, "fits" if mixed < head else "OVER"))
    print()
    print("The point is not the ratio, it is where the cliff is.  A growing")
    print("cache turns context length into a capacity limit, and capacity on a")
    print("bonded stack is set by the DRAM dies you can bond -- not something")
    print("the controller or the NoC can trade against.  A fixed state moves")
    print("most of that cost off the context axis entirely.")


if __name__ == "__main__":
    main()
