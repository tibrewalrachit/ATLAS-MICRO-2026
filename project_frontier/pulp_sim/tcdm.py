"""TCDM/HCI port contention (pessimistic, per guardrail).

Sourced: 32 banks x 8 B/clk = 256 B/clk cluster TCDM; RedMulE port 256-bit
(32 B/clk); NEUREKA HCI port modeled at 288-bit class ~ 32 B/clk; Snitch
SSR streams 8 B/clk each. DMA-in from the CUBE stack must share the same
banks. Pessimistic arbitration: available_bw = 256 B/clk; stall factor =
max(1, demanded/available).
"""
TCDM_BW = 256.0     # B/clk per cluster


def stall_factor(hwpe_bytes_per_clk, dma_bytes_per_clk, ssr_bytes_per_clk=0.0):
    demanded = hwpe_bytes_per_clk + dma_bytes_per_clk + ssr_bytes_per_clk
    return max(1.0, demanded / TCDM_BW)
