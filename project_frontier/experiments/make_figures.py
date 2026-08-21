"""Part XXVIII: publication figures. All data from frontier.csv, processed/
outputs, or direct analytical evaluation. Palette: validated reference set
(dataviz method), fixed slot order; single-hue sequential for magnitude;
no dual axes; thin marks; direct labels where they add reading value.
"""
import csv, json, os, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "models"))
sys.path.insert(0, os.path.join(HERE, "..", "analytical"))

from deepseek_v4_flash.dag import build_decode_dag as v4_dag
from minimax_m3.dag import build_decode_dag as m3_dag
from model import FabrikPoint, decode_step
from power import step_power

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})

def load_frontier():
    rows = []
    for r in csv.DictReader(open(os.path.join(RES, "frontier.csv"))):
        for k in ("batch", "context_tokens", "requested_bw_TBps", "peak_compute_TFLOPS",
                  "sram_MB", "TPS_per_user", "tokens_per_J", "cost_per_1m_output_tokens",
                  "total_power_W", "dram_utilization", "compute_utilization",
                  "arithmetic_intensity", "memory_capacity_GB"):
            r[k] = float(r[k])
        rows.append(r)
    return rows

ROWS = load_frontier()
V4 = [r for r in ROWS if r["model"] == "deepseek_v4_flash"
      and r["fits_capacity"] == "True" and r["feasible_capacity_mapping"] == "True"]
M3F = [r for r in ROWS if r["model_mode"] == "exact_fp8"
       and r["fits_capacity"] == "True" and r["feasible_capacity_mapping"] == "True"]

def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ---------- 1. decode roofline ----------
def fig01():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ai = np.logspace(-0.5, 2.2, 200)
    cases = [(24, 0.5, C[0]), (50, 1.0, C[1]), (100, 2.0, C[2])]
    for bw, pf, c in cases:
        roof = np.minimum(ai * bw * 1e12 * 0.8, pf * 1e15 * 0.85)
        ax.loglog(ai, roof / 1e12, color=c, lw=2,
                  label=f"{bw} TB/s, {pf} PF/s")
    for wl, ctx, m in [(v4_dag(131072), "V4-Flash 128K", "o"),
                       (v4_dag(1048576), "V4-Flash 1M", "s"),
                       (m3_dag(131072, "exact_fp8"), "M3 128K", "^"),
                       (m3_dag(1048576, "exact_fp8"), "M3 1M", "D")]:
        s = wl.summary()
        ax.scatter([s["ai"]], [s["flops"] / 1e9 / 1e3], color=INK, marker=m, s=28,
                   zorder=5)
        ax.annotate(ctx, (s["ai"], s["flops"] / 1e12), textcoords="offset points",
                    xytext=(6, -4), fontsize=7.5, color=INK2)
    ax.set_xlabel("arithmetic intensity (FLOP/byte), B=1 decode")
    ax.set_ylabel("achievable TFLOP/s")
    ax.set_title("Decode roofline: workloads vs Fabrik configurations")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "fig01_decode_roofline")

# ---------- 2. bandwidth x compute heatmaps ----------
def fig02():
    for model_rows, mname, tag in [(V4, "DeepSeek V4-Flash", "v4"), (M3F, "MiniMax M3 fp8", "m3")]:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
        bws = sorted({r["requested_bw_TBps"] for r in model_rows})
        pfs = sorted({r["peak_compute_TFLOPS"] for r in model_rows})
        for j, B in enumerate([1, 2, 4]):
            g = sel(model_rows, batch=B, context_tokens=131072.0)
            g = [r for r in g if r["sram_MB"] == 64 and r["memory_capacity_GB"] >= 256]
            z = np.full((len(pfs), len(bws)), np.nan)
            for r in g:
                if r["memory_capacity_GB"] == (256 if tag == "v4" else 512):
                    z[pfs.index(r["peak_compute_TFLOPS"]), bws.index(r["requested_bw_TBps"])] = r["TPS_per_user"]
            ax = axes[j]
            im = ax.imshow(z, origin="lower", aspect="auto", cmap="Blues",
                           vmin=0, vmax=np.nanmax(z))
            cs = ax.contour(z, levels=[500, 1000, 1500, 2000], colors=INK,
                            linewidths=0.9)
            ax.clabel(cs, fmt="%d", fontsize=6.5)
            ax.set_xticks(range(len(bws))); ax.set_xticklabels([int(b) for b in bws], fontsize=7)
            ax.set_yticks(range(len(pfs))); ax.set_yticklabels([f"{p/1000:g}" for p in pfs], fontsize=7)
            ax.set_xlabel("DRAM BW (TB/s)"); ax.grid(False)
            ax.set_title(f"B={B}", fontsize=9)
        axes[0].set_ylabel("compute (PFLOP/s)")
        fig.colorbar(im, ax=axes, shrink=0.85, label="TPS/user")
        fig.suptitle(f"{mname} @128K: TPS/user over bandwidth x compute", y=1.04)
        save(fig, f"fig02_heatmap_{tag}")

# ---------- 3/4. cost & energy frontiers ----------
def _frontier_scatter(rows, ykey, ylabel, name, refs, ylog=False, better="min"):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (B, c) in enumerate(zip([1, 2, 4], C[:3])):
        g = sel(rows, batch=float(B), context_tokens=131072.0)
        pts = sorted([(r["TPS_per_user"], r[ykey]) for r in g])
        # lower envelope (min cost) or upper (max tokens/J) per TPS bucket
        env = {}
        for x, y in pts:
            k = round(x / 100)
            if k not in env or ((y < env[k][1]) if better == "min" else (y > env[k][1])):
                env[k] = (x, y)
        e = sorted(env.values())
        ax.plot([p[0] for p in e], [p[1] for p in e], color=c, lw=2, label=f"B={B}")
    for label, x, y, c in refs:
        ax.scatter([x], [y], color=c, marker="*", s=90, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 2),
                    fontsize=7, color=INK2)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("TPS per user")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(name)
    return fig, ax

def fig03():
    refs = [("GPU cheapest route (V4-Flash, 306 t/s)", 306, 0.168, C[7]),
            ("DeepSeek API", 107, 0.66, C[7])]
    fig, ax = _frontier_scatter(V4, "cost_per_1m_output_tokens", "$ / 1M output tokens",
                                "V4-Flash @128K: cost vs interactivity", refs, ylog=True)
    save(fig, "fig03_cost_frontier")

def fig04():
    # GPU reference tokens/J: labeled ESTIMATE (H200-class node ~5.7kW serving
    # ~3500 aggregate t/s V4-Flash-class -> ~0.6 tok/J); marked in report.
    refs = [("GPU node estimate", 306, 0.6, C[7])]
    fig, ax = _frontier_scatter(V4, "tokens_per_J", "tokens / J",
                                "V4-Flash @128K: energy vs interactivity", refs,
                                better="max")
    save(fig, "fig04_energy_frontier")

# ---------- 5. TPS vs batch ----------
def fig05():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    hw = FabrikPoint("f50", 50, 1.0, 256, 64)
    hw2 = FabrikPoint("f100", 100, 2.0, 512, 128)
    for wlf, ctx, label, c in [(v4_dag, 131072, "V4-Flash, 50TB/s", C[0]),
                               (m3_dag, 131072, "M3 fp8, 100TB/s", C[1])]:
        xs, ys = [], []
        for B in (1, 2, 4):
            wl = wlf(ctx) if wlf is v4_dag else wlf(ctx, "exact_fp8")
            r = decode_step(wl, hw if wlf is v4_dag else hw2, B)
            xs.append(B); ys.append(r.tps_per_user)
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=5, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, 5),
                        fontsize=7, color=INK2)
    ax.axhline(1000, color=INK2, lw=0.8, ls="--"); ax.axhline(2000, color=INK2, lw=0.8, ls="--")
    ax.set_xticks([1, 2, 4]); ax.set_xlabel("batch size (users)"); ax.set_ylabel("TPS per user")
    ax.set_title("Interactivity vs batch @128K")
    ax.legend(fontsize=8)
    save(fig, "fig05_tps_vs_batch")

# ---------- 6. memory traffic decomposition ----------
CAT_GROUPS = [
    ("expert weights", ["routed_expert"], C[0]),
    ("common weights", ["dense_proj", "shared_expert", "router", "head", "residual_hc"], C[1]),
    ("KV", ["full_attn", "msa_attn", "local_attn", "csa_attn", "hca_attn"], C[2]),
    ("index", ["indexer", "msa_index"], C[3]),
    ("activations/other", ["norm", "other_vector"], C[4]),
]

def fig06():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cases = [("V4 32K", v4_dag(32768)), ("V4 128K", v4_dag(131072)), ("V4 1M", v4_dag(1048576)),
             ("M3 32K", m3_dag(32768, "exact_fp8")), ("M3 128K", m3_dag(131072, "exact_fp8")),
             ("M3 1M", m3_dag(1048576, "exact_fp8"))]
    xs = np.arange(len(cases))
    bottoms = np.zeros(len(cases))
    for label, cats, c in CAT_GROUPS:
        vals = []
        for _, wl in cases:
            t = wl.totals()
            vals.append(sum(t.get(k, {"weight": 0, "kv_r": 0, "kv_w": 0, "idx": 0, "act": 0})[f]
                            for k in cats for f in ("weight", "kv_r", "kv_w", "idx", "act")
                            if k in t) / 1e9)
        ax.bar(xs, vals, 0.62, bottom=bottoms, color=c, label=label,
               edgecolor="white", linewidth=1.2)
        bottoms += np.array(vals)
    for x, b in zip(xs, bottoms):
        ax.annotate(f"{b:.1f}", (x, b), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases], fontsize=8)
    ax.set_ylabel("GB per token (B=1)")
    ax.set_title("Decode memory-traffic decomposition")
    ax.legend(fontsize=7.5, ncol=2)
    save(fig, "fig06_traffic_decomposition")

# ---------- 7. latency decomposition ----------
def fig07():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    hw = FabrikPoint("f50", 50, 1.0, 256, 64)
    hw_m3 = FabrikPoint("f100", 100, 2.0, 512, 128)
    cases = [("V4 32K", v4_dag(32768), hw), ("V4 128K", v4_dag(131072), hw),
             ("V4 1M", v4_dag(1048576), hw),
             ("M3 32K", m3_dag(32768, "exact_fp8"), hw_m3),
             ("M3 128K", m3_dag(131072, "exact_fp8"), hw_m3),
             ("M3 1M", m3_dag(1048576, "exact_fp8"), hw_m3)]
    xs = np.arange(len(cases)); bottoms = np.zeros(len(cases))
    for label, cats, c in CAT_GROUPS:
        vals = []
        for _, wl, h in cases:
            r = decode_step(wl, h, 1)
            vals.append(sum(r.by_category.get(k, {"t": 0})["t"] for k in cats) * 1e3)
        ax.bar(xs, vals, 0.62, bottom=bottoms, color=c, label=label,
               edgecolor="white", linewidth=1.2)
        bottoms += np.array(vals)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases], fontsize=8)
    ax.set_ylabel("decode step ms (B=1)")
    ax.set_title("Latency decomposition (V4 on 50 TB/s; M3 on 100 TB/s)")
    ax.legend(fontsize=7.5, ncol=2)
    save(fig, "fig07_latency_decomposition")

# ---------- 8. power decomposition ----------
def fig08():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    hw = FabrikPoint("f50", 50, 1.0, 256, 64)
    labels = ["dram_array", "iface_3d", "compute", "sram", "noc", "controller", "static"]
    cases = [("V4 128K B=1", v4_dag(131072), 1), ("V4 128K B=4", v4_dag(131072), 4),
             ("V4 1M B=4", v4_dag(1048576), 4)]
    xs = np.arange(len(cases)); bottoms = np.zeros(len(cases))
    for i, comp in enumerate(labels):
        vals = []
        for _, wl, B in cases:
            r = decode_step(wl, hw, B)
            p = step_power(r, hw)
            vals.append(p["by_subsystem_J"][comp] / r.latency_s)
        ax.bar(xs, vals, 0.58, bottom=bottoms, color=C[i % len(C)], label=comp,
               edgecolor="white", linewidth=1.2)
        bottoms += np.array(vals)
    for x, b in zip(xs, bottoms):
        ax.annotate(f"{b:.0f} W", (x, b), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases], fontsize=8)
    ax.set_ylabel("average power (W)")
    ax.set_title("Power decomposition, Fabrik 50 TB/s / 1 PF")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig08_power_decomposition")

# ---------- 9. context scaling ----------
def fig09():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    hw = FabrikPoint("f50", 50, 1.0, 256, 64)
    hw_m3 = FabrikPoint("f100", 100, 2.0, 512, 128)
    ctxs = [8192, 32768, 131072, 524288, 1048576]
    for wlf, h, label, c in [(lambda t: v4_dag(t), hw, "V4-Flash (50 TB/s)", C[0]),
                             (lambda t: m3_dag(t, "exact_fp8"), hw_m3, "M3 fp8 (100 TB/s)", C[1])]:
        ys = [decode_step(wlf(t), h, 1).tps_per_user for t in ctxs]
        ax.semilogx(ctxs, ys, "-o", color=c, lw=2, ms=4, label=label)
    ax.axhline(1000, color=INK2, lw=0.8, ls="--")
    ax.axhline(2000, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(ctxs); ax.set_xticklabels(["8K", "32K", "128K", "512K", "1M"])
    ax.set_xlabel("context length"); ax.set_ylabel("TPS per user (B=1)")
    ax.set_title("Context scaling of interactivity")
    ax.legend(fontsize=8)
    save(fig, "fig09_context_scaling")

# ---------- 10. SRAM sensitivity ----------
def fig10():
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    srams = [8, 16, 32, 64, 128, 256]
    for bw, c in [(24, C[0]), (50, C[1]), (100, C[2])]:
        ys = [decode_step(v4_dag(131072), FabrikPoint("x", bw, 1.0, 256, s), 1).tps_per_user
              for s in srams]
        ax.semilogx(srams, ys, "-o", color=c, lw=2, ms=4, base=2, label=f"{bw} TB/s")
    ax.set_xticks(srams); ax.set_xticklabels(srams)
    ax.set_xlabel("SRAM (MB)"); ax.set_ylabel("TPS per user")
    ax.set_title("SRAM sensitivity (V4-Flash @128K, B=1)")
    ax.legend(fontsize=8)
    save(fig, "fig10_sram_sensitivity")

# ---------- 11/12. utilizations ----------
def fig11():
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    hw = FabrikPoint("f50", 50, 1.0, 256, 64)
    for wlf, label, c in [(lambda: v4_dag(131072), "V4-Flash", C[0]),
                          (lambda: m3_dag(131072, "exact_fp8"), "M3 fp8", C[1])]:
        xs, ys = [], []
        for B in (1, 2, 4):
            r = decode_step(wlf(), hw, B)
            xs.append(B); ys.append(r.compute_util * 100)
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=5, label=label)
    ax.set_xticks([1, 2, 4]); ax.set_xlabel("batch"); ax.set_ylabel("compute utilization (%)")
    ax.set_title("Compute utilization vs batch (50 TB/s, 1 PF)")
    ax.legend(fontsize=8)
    save(fig, "fig11_compute_util_vs_batch")

def fig12():
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    bws = [12, 24, 32, 40, 50, 60, 75, 100, 125, 150]
    for wlf, pf, label, c in [(lambda: v4_dag(131072), 1.0, "V4-Flash", C[0]),
                              (lambda: m3_dag(131072, "exact_fp8"), 2.0, "M3 fp8", C[1])]:
        ys = [decode_step(wlf(), FabrikPoint("x", b, pf, 512, 128), 1).dram_util * 100
              for b in bws]
        ax.plot(bws, ys, "-o", color=c, lw=2, ms=4, label=label)
    ax.set_xlabel("peak DRAM bandwidth (TB/s)"); ax.set_ylabel("DRAM utilization (%)")
    ax.set_title("Memory utilization vs bandwidth (B=1 @128K)")
    ax.legend(fontsize=8)
    save(fig, "fig12_mem_util_vs_bw")

# ---------- 13. validation ----------
def fig13():
    rows = list(csv.DictReader(open(os.path.join(RES, "processed", "validation_summary.csv"))))
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.17, [float(r["analytical_ms"]) for r in rows], 0.32,
           color=C[0], label="analytical", edgecolor="white")
    ax.bar(xs + 0.17, [float(r["cycle_ms"]) for r in rows], 0.32,
           color=C[1], label="ATLAS cycle sim", edgecolor="white")
    for i, r in enumerate(rows):
        ax.annotate(f'x{float(r["ratio"]):.2f}', (i, max(float(r["analytical_ms"]),
                     float(r["cycle_ms"]))), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{r["case"]}\nB={r["batch"]}' for r in rows], fontsize=7.5)
    ax.set_ylabel("decode step (ms)")
    ax.set_title("Analytical model vs cycle simulation")
    ax.legend(fontsize=8)
    save(fig, "fig13_validation")

# ---------- 14/15. min hardware ----------
def _min_hw_fig(target, name):
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "processed",
                                                        "min_hardware_for_targets.csv")))
            if r["feasible"] == "True" and int(r["target_tps"]) == target
            and r["mode"] in ("exact", "exact_fp8") and int(r["batch"]) in (1, 4)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    labels, bws, cols = [], [], []
    for r in sorted(rows, key=lambda r: (r["model"], int(r["context"]), int(r["batch"]))):
        ctx = {32768: "32K", 131072: "128K", 1048576: "1M"}[int(r["context"])]
        labels.append(f'{"V4" if "deepseek" in r["model"] else "M3"} {ctx} B={r["batch"]}')
        bws.append(float(r["min_bw_TBps"]))
        cols.append(C[0] if "deepseek" in r["model"] else C[1])
    xs = np.arange(len(labels))
    ax.bar(xs, bws, 0.62, color=cols, edgecolor="white")
    for x, b in zip(xs, bws):
        ax.annotate(f"{b:.0f}", (x, b), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("min DRAM BW (TB/s)")
    ax.set_title(f"Minimum bandwidth for {target} TPS/user "
                 f"(blue=V4-Flash, orange=M3-fp8; infeasible cases omitted)")
    save(fig, name)

def fig14(): _min_hw_fig(1000, "fig14_min_hw_1000tps")
def fig15(): _min_hw_fig(2000, "fig15_min_hw_2000tps")

if __name__ == "__main__":
    for f in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
              fig10, fig11, fig12, fig13, fig14, fig15]:
        try:
            f()
        except Exception as e:
            print(f"FIG FAIL {f.__name__}: {type(e).__name__}: {e}")
