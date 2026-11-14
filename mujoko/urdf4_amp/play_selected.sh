#!/usr/bin/env bash
set -euo pipefail
URDF4_MJ_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
URDF4_MJ_PYTHON="${URDF4_MJ_PYTHON:-/tmp/urdf4_validation_env/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$URDF4_MJ_ROOT"
exec "$URDF4_MJ_PYTHON" mujoko/urdf4_amp/sim2sim.py \
  --bundle artifacts/urdf4_mujoco_20261114 --follow "$@"
