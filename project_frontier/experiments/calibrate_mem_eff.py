"""Part XI: Ramulator-measured effective-bandwidth calibration.

Runs test_dram probes representing each decode access pattern on the ATLAS
HBDRAM config (16ch x 1024pin @500Mbps = 1 TB/s peak) and derives
pattern -> efficiency fractions for the analytical memory model.

Probe design: each pattern is measured in a short burst (dependency-latency
view, small N as it occurs once per layer) and steady state (large N,
approximating pipelined execution across layers/users). The analytical model
uses steady-state efficiency + explicit per-layer fixed latency.

Patterns:
  csa_row_gather:  576B rows, scattered (V4-Flash top-512 compressed KV)
  msa_block_gather:128-slot x 1KB blocks (M3 selected KV blocks)
  kv_slot_write:   scattered single-slot accesses (KV appends) [read proxy]
  index_scan:      full-range sequential coverage (V4 indexer / HCA scan)
  weight_stream:   large sequential GEMV weight stream (matrix test)
"""
import os, subprocess, csv, json, sys, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXE = os.path.join(ROOT, "simulator/build/bin/test_dram")
CFG = os.path.join(ROOT, "configs/architecture/chip/test_chip_16ch.yaml")
OUT_DIR = os.path.join(ROOT, "project_frontier/results/processed")
PEAK_GBPS = 1024.0   # 16ch x 64 GB/s


def run_probe(args):
    r = subprocess.run([EXE] + [str(a) for a in args], capture_output=True,
                       text=True, timeout=7200)
    out = r.stdout
    def grab(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else None
    return {"latency_s": grab(r"Latency: ([\d.e+-]+) s"),
            "bw_gbps": grab(r"BW: ([\d.e+-]+) GB/s"),
            "bw_util": grab(r"BW Util: ([\d.e+-]+)")}


ATTN_PROBES = [
    # name, slot_num, kv_bytes, block_size, n_blocks, mode
    ("csa_row_gather_burst",   262144, 576,  1,   512,   "burst"),
    ("csa_row_gather_steady",  262144, 576,  1,   16384, "steady"),
    ("msa_block_gather_burst", 262144, 1024, 128, 17,    "burst"),
    ("msa_block_gather_steady",262144, 1024, 128, 1024,  "steady"),
    ("kv_slot_scatter_steady", 262144, 576,  1,   16384, "steady"),
    ("index_scan_steady",      262144, 68,   2048, 128,  "steady"),   # full coverage, sequential-ish
    ("hca_scan_steady",        262144, 576,  4096, 64,   "steady"),   # contiguous compressed-KV scan
]
MATRIX_PROBES = [
    # name, M,K,N, tM,tK,tN, in_rm, out_rm, w_rm, elem, dataflow
    ("weight_stream_gemv", 1, 4096, 2048, 1, 512, 512, 1, 1, 1, 1, 0),
    ("weight_stream_big",  4, 4096, 8192, 4, 512, 512, 1, 1, 1, 1, 0),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name, slots, kvb, bs, nb, mode in ATTN_PROBES:
        res = run_probe([CFG, "attention", slots, kvb, bs, nb, 42])
        rows.append(dict(probe=name, mode=mode, kind="attention",
                         block_bytes=kvb * bs, n_blocks=nb, **res))
        print(f"{name}: bw={res['bw_gbps']:.1f} GB/s util={res['bw_util']:.3f} lat={res['latency_s']*1e6:.1f}us", flush=True)
    for name, M, K, N, tM, tK, tN, irm, orm, wrm, es, df in MATRIX_PROBES:
        res = run_probe([CFG, "matrix", M, K, N, tM, tK, tN, irm, orm, wrm, es, df])
        rows.append(dict(probe=name, mode="steady", kind="matrix",
                         block_bytes=K * N * es, n_blocks=1, **res))
        print(f"{name}: bw={res['bw_gbps']:.1f} GB/s util={res['bw_util']:.3f}", flush=True)
    with open(os.path.join(OUT_DIR, "mem_eff_calibration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # derive pattern efficiencies (steady-state)
    by = {r["probe"]: r for r in rows}
    eff = {
        "seq_stream": max(by["weight_stream_gemv"]["bw_util"], by["weight_stream_big"]["bw_util"]),
        "block_gather": by["msa_block_gather_steady"]["bw_util"],
        "random_gather": by["csa_row_gather_steady"]["bw_util"],
        "scan": max(by["index_scan_steady"]["bw_util"], by["hca_scan_steady"]["bw_util"]),
        "_burst_latency_csa_512rows_s": by["csa_row_gather_burst"]["latency_s"],
        "_burst_latency_msa_17blocks_s": by["msa_block_gather_burst"]["latency_s"],
        "_source": "ramulator test_dram on 16ch HBDRAM 1TB/s (test_chip_16ch)",
    }
    with open(os.path.join(OUT_DIR, "mem_eff_calibrated.json"), "w") as f:
        json.dump(eff, f, indent=2)
    print(json.dumps(eff, indent=2))


if __name__ == "__main__":
    main()
