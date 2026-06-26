#!/usr/bin/env bash
# Compile ONE kernel for several GPU architectures and dump every stage/pass for each.
# Lets you compare how the SAME kernel lowers differently per chip.
#
# Usage: ./dump_multi_chip.sh <file.py> <kernel> "<signature>" "<grid>" <out_subdir> [num_stages]
#
# Architectures covered (cuda:<sm>:<warpsize>):
#   sm_75  Turing      (T4, RTX 20xx)        - mma v2
#   sm_80  Ampere      (A100)                - mma v2, async copy (cp.async)
#   sm_86  Ampere      (RTX 3080, this box)  - mma v2
#   sm_90  Hopper      (H100)                - wgmma + warp specialization + TMA
# (sm_100 Blackwell needs ptxas-blackwell; add it if available.)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=$ROOT/.venv/bin/python
FILE=${1:?}; KERNEL=${2:?}; SIG=${3:?}; GRID=${4:?}; OUT=${5:?}; STAGES=${6:-3}

for ARCH in 75 80 86 90; do
  DUMP=$ROOT/learn_triton/dump/${OUT}/sm${ARCH}
  mkdir -p "$DUMP/stage_dump" "$DUMP/aot"
  echo "==== compiling for sm_${ARCH} ===="
  set +e
  PYTHONPATH=$ROOT/python \
  TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 \
  TRITON_DUMP_DIR=$DUMP/stage_dump MLIR_ENABLE_DUMP=1 \
  "$PY" "$ROOT/learn_triton/compile_driver.py" \
    "$FILE" "$KERNEL" "$SIG" "$ARCH" 4 "$STAGES" \
    > "$DUMP/mlir-pass-dump.log" 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "  sm_${ARCH}: FAILED (rc=$rc) — see $DUMP/mlir-pass-dump.log"; tail -3 "$DUMP/mlir-pass-dump.log"; continue
  fi
  "$PY" "$ROOT/learn_triton/split_pass_dump.py" "$DUMP/mlir-pass-dump.log" "$DUMP/mlir-pass-dump.split" >/dev/null
  echo "  sm_${ARCH}: ok -> $DUMP"
done
