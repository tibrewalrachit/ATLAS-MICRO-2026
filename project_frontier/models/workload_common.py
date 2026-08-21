"""Common decode-workload representation for Project Frontier.

Every operator in a single decode step (one token, one user) is described by an
`Op`. Builders in deepseek_v4_flash/ and minimax_m3/ emit per-layer op lists;
the analytical simulator composes batches, expert-reuse models, and hardware.

Scientific-standard notes (Part XXXVII):
- bytes are *achieved-traffic requests*, not bandwidth; efficiency factors are
  applied by the hardware model per access pattern, never here.
- weight_bytes are per-invocation streams for B=1; `shared_across_batch` marks
  weights that a batched execution reads once for all users (dense projections),
  vs per-user work (attention reads). Routed experts are handled by the MoE
  reuse model, not by this flag.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Engine(str, Enum):
    MATRIX = "matrix"      # GEMM/GEMV
    VECTOR = "vector"      # norms, softmax, elementwise, rope
    INDEX = "index"        # index scan / top-k / gather address generation
    MEMORY = "memory"      # pure data movement (KV write, cache mgmt)


class AccessPattern(str, Enum):
    SEQ_STREAM = "seq_stream"          # long contiguous weight/KV streams
    BLOCK_GATHER = "block_gather"      # gather of >=128B-contiguous blocks (MSA blocks, CSA rows)
    RANDOM_GATHER = "random_gather"    # fine-grained scattered reads
    SCAN = "scan"                      # full sequential scan of an index structure
    ON_CHIP = "on_chip"                # negligible DRAM traffic


@dataclass
class Op:
    name: str
    category: str            # dense_proj | shared_expert | routed_expert | router |
                             # indexer | csa_attn | hca_attn | local_attn | msa_index |
                             # msa_attn | full_attn | norm | residual_hc | head | other_vector
    engine: Engine
    flops: float = 0.0                 # per token per user
    weight_bytes: float = 0.0          # weight stream bytes (B=1)
    act_bytes: float = 0.0             # activation read+write bytes to DRAM (usually ~0; SRAM resident)
    kv_read_bytes: float = 0.0         # per user
    kv_write_bytes: float = 0.0        # per user
    index_read_bytes: float = 0.0      # per user (index structures)
    precision: str = "fp8"
    pattern: AccessPattern = AccessPattern.SEQ_STREAM
    shared_across_batch: bool = True   # True: weight stream amortizes over batch
    deps: List[str] = field(default_factory=list)
    count: int = 1                     # number of identical instances (layers collapsed later)
    notes: str = ""

    @property
    def total_bytes(self) -> float:
        return self.weight_bytes + self.act_bytes + self.kv_read_bytes + \
               self.kv_write_bytes + self.index_read_bytes


@dataclass
class DecodeWorkload:
    model: str
    mode: str                # exact | proxy variants
    context: int
    ops: List[Op] = field(default_factory=list)

    def totals(self):
        agg = {}
        for o in self.ops:
            c = agg.setdefault(o.category, dict(flops=0.0, weight=0.0, kv_r=0.0,
                                                kv_w=0.0, idx=0.0, act=0.0))
            c["flops"] += o.flops * o.count
            c["weight"] += o.weight_bytes * o.count
            c["kv_r"] += o.kv_read_bytes * o.count
            c["kv_w"] += o.kv_write_bytes * o.count
            c["idx"] += o.index_read_bytes * o.count
            c["act"] += o.act_bytes * o.count
        return agg

    def summary(self):
        t = self.totals()
        tot = {k: sum(c[k] for c in t.values()) for k in
               ["flops", "weight", "kv_r", "kv_w", "idx", "act"]}
        tot["bytes"] = tot["weight"] + tot["kv_r"] + tot["kv_w"] + tot["idx"] + tot["act"]
        tot["ai"] = tot["flops"] / tot["bytes"] if tot["bytes"] else 0.0
        return tot


# Precision byte sizes (per element); fp4 includes 1/16 block-scale overhead,
# fp8 block-128 scales add ~1/128 (ignored, <1%).
BYTES = {"fp4": 0.5 * (1 + 1 / 16), "fp8": 1.0, "bf16": 2.0, "fp32": 4.0}
