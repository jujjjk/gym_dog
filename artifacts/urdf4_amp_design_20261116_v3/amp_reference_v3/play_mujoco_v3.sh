#!/usr/bin/env bash
set -euo pipefail
amp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
amp_repo="${URDF4_REPO:-$(cd -- "$amp_dir/../../.." && pwd)}"
amp_python="${URDF4_VALIDATION_PYTHON:-/tmp/urdf4_validation_env/bin/python}"
amp_manifest="$amp_dir/manifest.json"
amp_clip="forward_p0500_p0000_p0000"
if [[ "${1:-}" == "--transitions" ]]; then
  shift
  amp_manifest="$amp_dir/transitions/manifest.json"
  amp_clip="start_stop_forward"
fi
export DISPLAY="${DISPLAY:-:1}"
cd "$amp_repo"
exec "$amp_python" tools/urdf4_amp/physical_v3.py \
  --design "$amp_repo/artifacts/urdf4_amp_design_20261116_v3/plant" \
  --spec "$amp_manifest" --motion "$amp_clip" --viewer "$@"
