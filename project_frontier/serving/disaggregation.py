"""Prefill/decode disaggregation model (Part XXII).

T_request = T_prefill_GPU + T_handoff + N_output x T_decode_Fabrik
Handoff = KV_state_bytes / interconnect_BW + latency + session metadata.
GPU prefill throughput is an external parameter (labeled), default:
8xB200-class node, 55k tok/s effective prefill for a 13B-active MoE (assumed;
swept in sensitivity).
"""
def handoff_time(kv_bytes, interconnect_gbps, latency_s, metadata_bytes=2e6):
    return (kv_bytes + metadata_bytes) / (interconnect_gbps * 1e9) + latency_s

def request_time(prefill_tokens, prefill_tps_gpu, kv_bytes,
                 interconnect_gbps, latency_s, n_output, decode_step_s):
    tp = prefill_tokens / prefill_tps_gpu
    th = handoff_time(kv_bytes, interconnect_gbps, latency_s)
    td = n_output * decode_step_s
    ttft = tp + th + decode_step_s
    return {"T_prefill_s": tp, "T_handoff_s": th, "T_decode_s": td,
            "TTFT_s": ttft, "T_request_s": tp + th + td,
            "handoff_frac_of_request": th / (tp + th + td)}

INTERCONNECT_GRID_GBPS = [64, 128, 256, 512, 1000]
LATENCY_GRID_S = [10e-6, 5e-6, 1e-6, 500e-9, 100e-9]
