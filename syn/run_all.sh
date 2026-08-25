#!/usr/bin/env bash
#
# Synthesise and time-analyse every ATLAS block, and write a summary table.
#
#   usage: syn/run_all.sh [period_ns] [tech]
#
# Output: sta/out/summary_<tech>.txt
#
# Blocks whose area is dominated by compiled SRAM macros (atlas_buffer, and
# atlas_core which contains one) are excluded: their timing comes from the
# memory compiler's Liberty, not from standard-cell synthesis, and mapping a
# 4 MB scratchpad to flip-flops would report an area that means nothing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERIOD_NS="${1:-1.0}"
TECH="${2:-asap7}"
PERIOD_PS=$(python3 -c "print(int(round($PERIOD_NS*1000)))")

OUT="$ROOT/sta/out/summary_${TECH}.txt"
mkdir -p "$ROOT/sta/out"

# name : parameter overrides
BLOCKS=(
  "atlas_dot_unit:"
  "atlas_fp32_add:"
  "atlas_fp32_mul:"
  "atlas_exp_f32:"
  "atlas_recip_f32:"
  "atlas_rsqrt_f32:"
  "atlas_silu_f32:"
  "atlas_vec_lane:"
  "atlas_sfu:"
  "atlas_pe:"
  "atlas_rmsnorm:"
  "atlas_moe_router:"
  "atlas_hbdram_ctrl:"
  "atlas_dma:"
  "atlas_noc_router:"
  "atlas_matrix_unit:ROWS=2 COLS=2"
  "atlas_vector_unit:VEC_N=8 SFU_RATIO=8"
)

{
  echo "ATLAS synthesis and static timing summary"
  echo "technology : $TECH        target period : ${PERIOD_NS} ns"
  echo "tools      : yosys $(yosys -V 2>/dev/null | head -1 | awk '{print $2}'), OpenSTA $(${STA_BIN:-/home/user/OpenSTA/build/sta} -version 2>/dev/null)"
  echo
  printf "%-22s %8s %12s %10s %10s %12s %s\n" \
    "block" "cells" "area_um2" "setup_ns" "hold_ns" "power_mW" "params"
  printf "%s\n" "----------------------------------------------------------------------------------------------"
} > "$OUT"

for entry in "${BLOCKS[@]}"; do
  name="${entry%%:*}"
  params="${entry#*:}"

  if ! "$ROOT/syn/run_synth.sh" "$name" "$PERIOD_PS" "$TECH" "$params" > /dev/null 2>&1; then
    printf "%-22s %8s %12s %10s %10s %12s %s\n" "$name" "SYNTH-FAIL" "-" "-" "-" "-" "$params" >> "$OUT"
    echo "  $name : synthesis failed"
    continue
  fi

  cells=$(grep -oP 'Number of cells:\s+\K[0-9]+' "$ROOT/syn/out/${name}_${TECH}.area.txt" | tail -1)
  area=$(grep -oP "Chip area for module.*: \K[0-9.]+" "$ROOT/syn/out/${name}_${TECH}.area.txt" | tail -1)

  "$ROOT/sta/run_sta.sh" "$name" "$PERIOD_NS" "$TECH" > /dev/null 2>&1
  rpt="$ROOT/sta/out/${name}_${TECH}.rpt"
  setup=$(grep -oP 'setup WNS : \K-?[0-9.]+' "$rpt" | tail -1)
  hold=$(grep -oP 'hold  WNS : \K-?[0-9.]+' "$rpt" | tail -1)
  # report_power's Total row: internal, switching, leakage, total
  power=$(grep -E '^Total' "$rpt" | tail -1 | awk '{print $5}')

  printf "%-22s %8s %12s %10s %10s %12s %s\n" \
    "$name" "${cells:--}" "${area:--}" "${setup:--}" "${hold:--}" "${power:--}" "$params" >> "$OUT"
  echo "  $name : cells=${cells:--} area=${area:--} setup=${setup:--} ns"
done

echo >> "$OUT"
echo "Excluded: atlas_buffer and atlas_core contain compiled SRAM macros, whose" >> "$OUT"
echo "timing belongs to the memory compiler's Liberty rather than to standard-cell" >> "$OUT"
echo "synthesis; mapping a 4 MB scratchpad to flip-flops would report a meaningless" >> "$OUT"
echo "area.  atlas_chip at full geometry is roughly 90 million gates and does not" >> "$OUT"
echo "elaborate on a 16 GB machine." >> "$OUT"

echo
cat "$OUT"
