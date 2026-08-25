#!/usr/bin/env bash
#
# Find the achievable clock frequency of an ATLAS block.
#
#   usage: sta/run_fmax.sh <top-module> [tech] [lo_ns] [hi_ns] [steps]
#
# Synthesis is re-run at every trial period, because ABC optimises against the
# target it is given -- sweeping the SDC alone over one netlist would report a
# frequency the design was never actually built for.
#
# Writes sta/out/<top>_<tech>.fmax.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP="${1:?usage: run_fmax.sh <top> [tech] [lo_ns] [hi_ns] [steps]}"
TECH="${2:-asap7}"
LO="${3:-0.6}"
HI="${4:-2.0}"
STEPS="${5:-7}"

OUT="$ROOT/sta/out/${TOP}_${TECH}.fmax.txt"
mkdir -p "$ROOT/sta/out"

echo "# Fmax sweep: $TOP on $TECH" | tee "$OUT"
echo "# period_ns  slack_ns  met  cells  area_um2" | tee -a "$OUT"

best=""
for k in $(seq 0 $((STEPS-1))); do
  p=$(python3 -c "print(f'{$LO + ($HI-$LO)*$k/max(1,$STEPS-1):.3f}')")
  ps=$(python3 -c "print(int(round($p*1000)))")

  "$ROOT/syn/run_synth.sh" "$TOP" "$ps" "$TECH" > /dev/null 2>&1
  cells=$(grep -oP 'Number of cells:\s+\K[0-9]+' "$ROOT/syn/out/${TOP}_${TECH}.area.txt" | tail -1)
  area=$(grep -oP "Chip area for module.*: \K[0-9.]+" "$ROOT/syn/out/${TOP}_${TECH}.area.txt" | tail -1)

  set +e
  "$ROOT/sta/run_sta.sh" "$TOP" "$p" "$TECH" > /dev/null 2>&1
  set -e
  slack=$(grep -oP 'setup WNS : \K-?[0-9.]+' "$ROOT/sta/out/${TOP}_${TECH}.rpt" | tail -1)
  met=$(python3 -c "print('yes' if $slack >= 0 else 'no')")
  [[ "$met" == "yes" && -z "$best" ]] && best="$p"

  printf "%10s  %8s  %3s  %7s  %s\n" "$p" "$slack" "$met" "$cells" "$area" | tee -a "$OUT"
done

if [[ -n "$best" ]]; then
  f=$(python3 -c "print(f'{1000.0/$best:.0f}')")
  echo "# fastest period meeting timing: ${best} ns  (${f} MHz)" | tee -a "$OUT"
else
  echo "# no swept period met timing; widen the range" | tee -a "$OUT"
fi
