#!/usr/bin/env bash
#
# Run every ATLAS RTL testbench and report a pass/fail summary.
#
#   usage: tb/run_all.sh
#
# Each testbench regenerates its own stimulus from the Python golden model
# first, so a change to the reference and a change to the RTL cannot silently
# drift apart.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OBJ="${OBJ_DIR:-/tmp/atlas_tb}"
mkdir -p "$OBJ"

RTL_ALL=$(find rtl -name '*.sv' | sort | tr '\n' ' ')
VFLAGS="--binary --timing -j $(nproc) -Wno-DECLFILENAME -Wno-UNUSEDPARAM -Wno-WIDTHTRUNC -Irtl/include"

pass=0; fail=0
declare -a FAILED

run_tb () {
  local name="$1"; shift
  local top="$1";  shift
  local srcs="$*"

  printf "%-20s " "$name"
  rm -rf "$OBJ/$name"
  if ! verilator $VFLAGS --top-module "$top" -o "$name" --Mdir "$OBJ/$name" $srcs \
       > "$OBJ/$name.build.log" 2>&1; then
    echo "BUILD FAIL   (see $OBJ/$name.build.log)"
    fail=$((fail+1)); FAILED+=("$name"); return
  fi
  local out
  out=$("$OBJ/$name/$name" 2>&1)
  echo "$out" > "$OBJ/$name.run.log"
  if echo "$out" | grep -q ": PASS"; then
    echo "PASS  $(echo "$out" | grep ': PASS' | sed 's/.*PASS//')"
    pass=$((pass+1))
  else
    echo "FAIL         (see $OBJ/$name.run.log)"
    echo "$out" | tail -3 | sed 's/^/                     /'
    fail=$((fail+1)); FAILED+=("$name")
  fi
}

echo "Regenerating stimulus from the golden model"
python3 tb/golden/gen_dot_vectors.py  tb/sv/dot_vectors.txt   > /dev/null
python3 tb/golden/gen_fadd_vectors.py tb/sv/fadd_vectors.txt  > /dev/null
python3 tb/golden/gen_gemm_vectors.py tb/sv/gemm_vectors.txt  > /dev/null
python3 tb/golden/gen_moe_vectors.py  tb/sv/moe_vectors.txt   > /dev/null
python3 tb/golden/gen_kda_vectors.py  tb/sv/kda_vectors.txt   > /dev/null
echo

echo "Testbenches"
echo "-----------------------------------------------------------------------"
run_tb tb_dot_unit    tb_dot_unit    tb/sv/tb_dot_unit.sv    $RTL_ALL
run_tb tb_csa_tree    tb_csa_tree    tb/sv/tb_csa_tree.sv    rtl/fp/atlas_csa_tree.sv
run_tb tb_fp32_add    tb_fp32_add    tb/sv/tb_fp32_add.sv    $RTL_ALL
run_tb tb_fp32_mul    tb_fp32_mul    tb/sv/tb_fp32_mul.sv    $RTL_ALL
run_tb tb_matrix_unit tb_matrix_unit tb/sv/tb_matrix_unit.sv $RTL_ALL
run_tb tb_rmsnorm     tb_rmsnorm     tb/sv/tb_rmsnorm.sv     $RTL_ALL
run_tb tb_moe_router  tb_moe_router  tb/sv/tb_moe_router.sv  $RTL_ALL
run_tb tb_hbdram      tb_hbdram      "tb/sv/tb_hbdram.sv tb/sv/hbdram_model.sv $RTL_ALL"
run_tb tb_noc_mesh    tb_noc_mesh    tb/sv/tb_noc_mesh.sv    $RTL_ALL
run_tb tb_kda         tb_kda         tb/sv/tb_kda.sv         $RTL_ALL

echo "-----------------------------------------------------------------------"
echo "$pass passed, $fail failed"
if [ $fail -ne 0 ]; then
  echo "failed: ${FAILED[*]}"
  exit 1
fi
