#!/usr/bin/env bash
# Compile a Triton kernel ahead-of-time and capture every compiler dump.
# No torch / no running on the GPU required — this only exercises the COMPILER.
#
# Usage:
#   ./compile_and_dump.sh <kernel.py> <kernel_name> "<signature>" "<grid>" <out_subdir>
#
# Example (vector add):
#   ./compile_and_dump.sh vec_add.py add_kernel "*fp32:16, *fp32:16, *fp32:16, i32, 1024" "1024,1,1" vecadd
#
# Signature grammar: comma-separated arg types in declaration order.
#   *fp32      pointer to fp32        i32 / i64 / fp32 ...  scalar
#   *fp32:16   pointer 16-byte aligned (divisibility hint -> better vectorization)
#   1024       a constexpr value (tl.constexpr args are passed as literals)
set -euo pipefail

ROOT=/LocalRun/jiangzhe.zhao/triton
PY=$ROOT/.venv/bin/python

KERNEL_FILE=${1:?kernel file}
KERNEL_NAME=${2:?kernel name}
SIGNATURE=${3:?signature}
GRID=${4:?grid}
OUT=${5:?out subdir}

DUMP=$ROOT/learn_triton/dump/$OUT
mkdir -p "$DUMP/stage_dump" "$DUMP/aot"

PYTHONPATH=$ROOT/python \
TRITON_ALWAYS_COMPILE=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=$DUMP/stage_dump \
MLIR_ENABLE_DUMP=1 \
"$PY" "$ROOT/python/triton/tools/compile.py" \
  --kernel-name "$KERNEL_NAME" \
  --signature "$SIGNATURE" \
  --grid "$GRID" \
  --out-name "$KERNEL_NAME" \
  --out-path "$DUMP/aot/$KERNEL_NAME" \
  "$KERNEL_FILE" \
  > "$DUMP/mlir-pass-dump.log" 2>&1

"$PY" "$ROOT/learn_triton/split_pass_dump.py" \
  "$DUMP/mlir-pass-dump.log" "$DUMP/mlir-pass-dump.split" >/dev/null

echo "Dumps written under: $DUMP"
echo "  stage_dump/<hash>/$KERNEL_NAME.{ttir,ttgir,llir,ptx,sass,cubin}  <- the 5 stage boundaries"
echo "  mlir-pass-dump.log                                               <- IR after every MLIR pass (full)"
echo "  mlir-pass-dump.split/NNN_*.mlir                                  <- same, one file per pass"
