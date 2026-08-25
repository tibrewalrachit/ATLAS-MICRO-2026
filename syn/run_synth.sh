#!/usr/bin/env bash
#
# Synthesise one ATLAS RTL module to gates with yosys.
#
#   usage: syn/run_synth.sh <top-module> [period_ps] [tech]
#
#   tech = asap7 (default, 7nm 0.7V TT) | sky130 (130nm 1.8V TT)
#
# Produces syn/out/<top>_<tech>.v (a flat gate netlist for OpenSTA) and
# syn/out/<top>_<tech>.area.txt (cell counts and area from `stat`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP="${1:?usage: run_synth.sh <top> [period_ps] [tech]}"
PERIOD_PS="${2:-1000}"
TECH="${3:-asap7}"

case "$TECH" in
  asap7)  LIB="$ROOT/syn/lib/asap7_merged.lib"; CONSTR="$ROOT/syn/lib/asap7.constr"  ;;
  sky130) LIB="$ROOT/syn/lib/sky130hd_tt.lib";  CONSTR="$ROOT/syn/lib/sky130.constr" ;;
  *) echo "unknown tech: $TECH" >&2; exit 1 ;;
esac

OUT="$ROOT/syn/out"
mkdir -p "$OUT"
NETLIST="$OUT/${TOP}_${TECH}.v"
AREA="$OUT/${TOP}_${TECH}.area.txt"
LOG="$OUT/${TOP}_${TECH}.synth.log"

# sky130 liberty declares time in ns; yosys' abc -D always wants picoseconds.
ABC_D="$PERIOD_PS"

# Passing -constr switches ABC to the script that also runs "buffer; upsize;
# dnsize".  Without it ABC maps to gates but never fixes drive strength, so a
# minimum-size cell can end up driving a 30-way fanout net and dominate the
# critical path.  This is the closest the open flow gets to the drive fixing a
# commercial synthesiser does by default.

SRC=$(find "$ROOT/rtl" -name '*.sv' | sort | tr '\n' ' ')

yosys -l "$LOG" -p "
  read_verilog -sv -I $ROOT/rtl/include $SRC
  hierarchy -check -top $TOP
  synth -top $TOP -flatten
  dfflibmap -liberty $LIB
  abc -liberty $LIB -constr $CONSTR -D $ABC_D
  setundef -zero
  splitnets -ports
  opt_clean -purge
  write_verilog -noattr -noexpr $NETLIST
  tee -o $AREA stat -liberty $LIB
" > /dev/null

echo "netlist : $NETLIST"
echo "area    : $AREA"
grep -E "Chip area|Number of cells" "$AREA" | tail -3
