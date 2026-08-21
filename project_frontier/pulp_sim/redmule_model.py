"""RedMulE closed-form cycle model (fidelity: paper-validated closed form).

Architecture (sourced, params/pulp_tile.yaml): W=12 parallel rows x H=4
chained CEs x (PIPE_REGS+1)=4 pipeline slots -> 48 FMA/cycle peak; each row
consumes K in chunks of H*(P+1)=16.

Cycle model for Z[M,N] = X[M,K] . W[K,N]:
  rows_active = min(M', W)   with the OPERAND-SWAP mapping for skinny M:
  for M < W the kernel computes Z^T = W^T . X^T so the streamed large
  dimension occupies the row dimension (M' = N). SSR/streamer-fed, so the
  swap costs no extra traffic (weights stream from TCDM either way).
  cycles = ceil(M'/W) * ceil(K/16) * 16 * N' / (H*(P+1)) + FILL
         = padded_MACs / 48 + FILL
Validation anchor: paper reports 99.4% utilization on 96x96x96 GEMM
(MACs/48 = 18,432 cycles; 0.6% overhead -> FILL ~= 111 cycles).
"""
import math

W_ROWS = 12
H_CE = 4
PIPE = 3
K_CHUNK = H_CE * (PIPE + 1)          # 16
PEAK_MAC = W_ROWS * H_CE             # 48 FMA/cycle (FP16; FP8 casts to FP16)
FILL_CYCLES = 111                    # derived from paper's 99.4% @ 96^3


def gemm_cycles(M, N, K, allow_swap=True):
    if allow_swap and M < W_ROWS <= N:
        M, N = N, M                  # operand swap for skinny-M decode GEMV
    m_pad = math.ceil(M / W_ROWS) * W_ROWS
    k_pad = math.ceil(K / K_CHUNK) * K_CHUNK
    return m_pad * k_pad * N / PEAK_MAC + FILL_CYCLES


def utilization(M, N, K, allow_swap=True):
    return (M * N * K / PEAK_MAC) / gemm_cycles(M, N, K, allow_swap)
