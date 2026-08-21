"""NEUREKA closed-form cycle model (fidelity: architecture-derived closed
form; NE16-family bit-serial dataflow).

Sourced geometry (params/pulp_tile.yaml): TP_IN=32 input channels x
TP_OUT=32 output channels per PE-cycle-group, up to 36 PEs over SPATIAL
points, weights processed bit-serially over Qw cycles (Qw = weight bits,
2..8).

KEY DATAFLOW FACT (this is why the cycle model matters): the PE array
parallelizes SPATIAL positions. A decode GEMV (one token) has spatial=B
(batch), so at B=1 only ONE PE is active:
  effective MACs/cycle = TP_IN * TP_OUT * min(B, NUM_PE) / Qw
  B=1, Qw=4 (int4/fp4-decoded): 32*32*1/4 = 256 MAC/cycle  (NOT 9,216)
The 36-PE array only pays off at batch >= 36 or for conv-shaped work.

cycles(GEMV K,N; B; Qw) = ceil(K/32) * ceil(N/32) * Qw * ceil(B/NUM_PE)
                          * 32-chunk latency (1c/group) + LOAD_OVERHEAD
Weight traffic: K*N*Qw/8 bytes (+ int8 activations). FP4-e2m1 experts
require decode-to-int or int4-block requant (ASSUMPTION P8).
"""
import math

TP_IN = 32
TP_OUT = 32
NUM_PE = 36
LOAD_OVERHEAD = 64        # streamer/job setup per invocation (ASSUMED, ranged)


def gemv_cycles(K, N, batch=1, qw=4):
    groups = math.ceil(K / TP_IN) * math.ceil(N / TP_OUT)
    spatial_passes = math.ceil(batch / NUM_PE)
    return groups * qw * spatial_passes + LOAD_OVERHEAD


def effective_macs_per_cycle(batch=1, qw=4):
    return TP_IN * TP_OUT * min(batch, NUM_PE) / qw
