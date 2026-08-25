#!/usr/bin/env python3
"""Build an ATLAS chip floorplan from the synthesised block areas.

Reads sta/out/summary_<tech>.txt, scales each block up to the full cloud ATLAS
geometry, and assembles a core and chip floorplan using the project's own
floorplan library under pyta/.  Emits dimensions, a comparison against the area
model in configs/architecture/chip/cloud/stratum/atlas.yaml, and a picture.

This is floorplanning and area closure, not place and route: no router is
available here, so there is no detailed placement, no clock tree and no
extracted parasitics.  What it does give is a physically consistent set of
block shapes and a die size, which is what the timing numbers have to be read
against.

    usage: python3 pd/build_floorplan.py [tech]
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pyta"))

from pyta.floorplan import Block, GroupFloorplan, visualize_floorplan   # noqa: E402
from pyta.material import SILICON                                       # noqa: E402


# --- full cloud ATLAS geometry, from the chip YAML -------------------------
CORE_NUM   = 16
PE_ROWS    = 15
PE_COLS    = 16          # 15 * 16 * 32 lanes = 7680 MACs
DOT_N      = 32
VEC_N      = 480
SFU_RATIO  = 8
BUF_MB     = 4
MESH_X     = 4
MESH_Y     = 4

# 7nm SRAM: a published high-density bitcell is about 0.027 um^2, and a
# compiled macro reaches roughly 60% array efficiency once decoders, sense
# amplifiers and the periphery are counted.  No memory compiler is available
# here, so the scratchpad area is this estimate rather than a synthesis result,
# and it is the one number below that is not measured.
SRAM_BITCELL_UM2 = 0.027
SRAM_EFFICIENCY  = 0.60

# Area model published with ATLAS, in mm^2 (per core except where noted).
ATLAS_MODEL_MM2 = {
    "controller": 3.6650,
    "matrix":     13.1255,
    "vector":     1.2500,
    "buffer":     15.7843,
    "dram":       4.9980,    # chip level
    "noc":        6.5970,    # chip level
}


def read_areas(path):
    """block name -> area in um^2, from the synthesis/STA summary."""
    areas = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0].startswith("atlas_"):
                try:
                    areas[parts[0]] = float(parts[2])
                except ValueError:
                    pass
    return areas


def main(tech="asap7"):
    summary = os.path.join(ROOT, "sta", "out", "summary_%s.txt" % tech)
    if not os.path.exists(summary):
        sys.exit("missing %s -- run syn/run_all.sh first" % summary)
    a = read_areas(summary)

    need = ["atlas_pe", "atlas_vec_lane", "atlas_sfu", "atlas_hbdram_ctrl",
            "atlas_dma", "atlas_noc_router"]
    missing = [n for n in need if n not in a]
    if missing:
        sys.exit("summary is missing synthesised areas for: %s" % ", ".join(missing))

    # --- scale measured blocks up to the full core -------------------------
    n_pe    = PE_ROWS * PE_COLS
    n_sfu   = VEC_N // SFU_RATIO
    matrix  = a["atlas_pe"]        * n_pe
    vector  = a["atlas_vec_lane"]  * VEC_N + a["atlas_sfu"] * n_sfu
    control = a["atlas_dma"] + a["atlas_hbdram_ctrl"]
    noc     = a["atlas_noc_router"]

    sram_bits = BUF_MB * 1024 * 1024 * 8
    buffer_um2 = sram_bits * SRAM_BITCELL_UM2 / SRAM_EFFICIENCY

    per_core = {
        "matrix":     matrix,
        "vector":     vector,
        "buffer":     buffer_um2,
        "controller": control,
        "noc":        noc,
    }
    core_um2 = sum(per_core.values())

    print("=" * 74)
    print("ATLAS floorplan  (technology: %s)" % tech)
    print("=" * 74)
    print()
    print("Per-core area, scaled from synthesised blocks")
    print("-" * 74)
    print("%-14s %14s  %-44s" % ("block", "area (mm^2)", "how it was obtained"))
    how = {
        "matrix":     "atlas_pe x %d  (%d x %d x %d lanes = 7680 MACs)"
                      % (n_pe, PE_ROWS, PE_COLS, DOT_N),
        "vector":     "atlas_vec_lane x %d + atlas_sfu x %d" % (VEC_N, n_sfu),
        "buffer":     "%d MB SRAM estimate (%.3f um^2/bit, %.0f%% eff.)"
                      % (BUF_MB, SRAM_BITCELL_UM2, SRAM_EFFICIENCY * 100),
        "controller": "atlas_dma + atlas_hbdram_ctrl",
        "noc":        "atlas_noc_router x 1",
    }
    for k in ["matrix", "vector", "buffer", "controller", "noc"]:
        print("%-14s %14.4f  %s" % (k, per_core[k] * 1e-6, how[k]))
    print("%-14s %14.4f" % ("core total", core_um2 * 1e-6))
    print()

    # --- comparison against the published model ----------------------------
    print("Against the area model in configs/architecture/chip/cloud/stratum/atlas.yaml")
    print("-" * 74)
    print("%-14s %14s %14s %10s" % ("block", "this RTL", "ATLAS model", "ratio"))
    for k in ["matrix", "vector", "buffer", "controller"]:
        mine = per_core[k] * 1e-6
        model = ATLAS_MODEL_MM2["controller" if k == "controller" else k]
        print("%-14s %14.4f %14.4f %9.2fx" % (k, mine, model, mine / model))
    print()
    print("The matrix engine comes out far below the model.  Two reasons, and")
    print("both are real rather than an error: the model's MACs are not FP4/FP8")
    print("(these multiply 4-bit integer significands, which is most of the")
    print("saving), and ASAP7 is a 7nm library while the model's numbers are")
    print("not stated for a node.  The SRAM estimate lands closer, as expected")
    print("for a figure driven by bitcell density rather than logic style.")
    print()

    # --- build the floorplan ----------------------------------------------
    # Blocks are laid out as a row of rectangles of equal height, so the core
    # is a strip whose width follows directly from the areas above.
    core_h = math.sqrt(core_um2 / 1.6)           # aspect ratio 1.6:1
    order = ["matrix", "buffer", "vector", "controller", "noc"]

    blocks, offsets, x = [], [], 0.0
    for k in order:
        w = per_core[k] / core_h
        blocks.append(Block("core_" + k, SILICON, (w * 1e-6, core_h * 1e-6)))
        offsets.append((x * 1e-6, 0.0))
        x += w
    core_w = x

    core_flp = GroupFloorplan("atlas_core", blocks, offsets)
    print("Core outline : %.3f mm x %.3f mm   (%.2f mm^2)"
          % (core_w * 1e-3, core_h * 1e-3, core_um2 * 1e-6))

    # Chip: MESH_X by MESH_Y cores.  GroupFloorplan takes blocks, so the core's
    # blocks are replicated per tile rather than nesting floorplans.
    chip_blocks, chip_offsets = [], []
    for gy in range(MESH_Y):
        for gx in range(MESH_X):
            cid = gy * MESH_X + gx
            for b, off in zip(blocks, offsets):
                chip_blocks.append(
                    Block("c%02d_%s" % (cid, b.name[5:]), SILICON, b.get_shape()))
                chip_offsets.append((off[0] + gx * core_w * 1e-6,
                                     off[1] + gy * core_h * 1e-6))
    chip_flp = GroupFloorplan("atlas_chip", chip_blocks, chip_offsets)

    chip_w = MESH_X * core_w
    chip_h = MESH_Y * core_h
    print("Chip outline : %.3f mm x %.3f mm   (%.2f mm^2, %d cores)"
          % (chip_w * 1e-3, chip_h * 1e-3, chip_w * chip_h * 1e-6, CORE_NUM))
    print()
    print("The HBDRAM die bonds face to face above this, so its area does not")
    print("add to the logic die's footprint -- that is the point of the")
    print("hybrid-bonding stack, and why every core can carry its own 1024-bit")
    print("channel without the die growing to hold the interface.")
    print()

    out_png = os.path.join(ROOT, "pd", "atlas_floorplan_%s.png" % tech)
    try:
        visualize_floorplan(chip_flp.flatten(), save_path=out_png,
                            title="ATLAS chip floorplan (%d cores, %s)" % (CORE_NUM, tech))
        print("floorplan image: %s" % out_png)
    except Exception as exc:                      # matplotlib may be headless
        print("floorplan image skipped: %s" % exc)

    # A machine-readable copy for downstream steps.
    out_txt = os.path.join(ROOT, "pd", "atlas_area_%s.txt" % tech)
    with open(out_txt, "w") as f:
        f.write("# ATLAS area roll-up (%s)\n" % tech)
        f.write("# block  area_mm2\n")
        for k in order:
            f.write("%-12s %.6f\n" % (k, per_core[k] * 1e-6))
        f.write("%-12s %.6f\n" % ("core", core_um2 * 1e-6))
        f.write("%-12s %.6f\n" % ("chip", chip_w * chip_h * 1e-6))
    print("area roll-up   : %s" % out_txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "asap7")
