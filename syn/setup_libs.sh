#!/usr/bin/env bash
#
# Stage the standard-cell Liberty files the flow needs.
#
# ASAP7 and sky130hd both ship with OpenSTA, so they are copied out of an
# OpenSTA checkout rather than vendored here.  ASAP7 splits its cells across
# three libraries (combinational, sequential, inverter/buffer) and yosys'
# mapping passes each take a single file, so they are merged.
#
#   usage: syn/setup_libs.sh [path-to-OpenSTA]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STA_SRC="${1:-${OPENSTA_DIR:-/home/user/OpenSTA}}"
LIB="$ROOT/syn/lib"

[[ -d "$STA_SRC/test" ]] || {
  echo "OpenSTA sources not found at $STA_SRC" >&2
  echo "pass the path, or set OPENSTA_DIR" >&2
  exit 1
}

mkdir -p "$LIB"

for f in asap7_simple asap7_seq asap7_invbuf; do
  [[ -f "$LIB/$f.lib" ]] || zcat "$STA_SRC/test/$f.lib.gz" > "$LIB/$f.lib"
done
[[ -f "$LIB/sky130hd_tt.lib" ]] || \
  cp "$STA_SRC/test/sky130hd/sky130_fd_sc_hd__tt_025C_1v80.lib" "$LIB/sky130hd_tt.lib"

[[ -f "$LIB/asap7_merged.lib" ]] || \
  python3 "$ROOT/syn/scripts/merge_liberty.py" "$LIB/asap7_merged.lib" \
    "$LIB/asap7_simple.lib" "$LIB/asap7_seq.lib" "$LIB/asap7_invbuf.lib"

echo "liberty ready in $LIB"
