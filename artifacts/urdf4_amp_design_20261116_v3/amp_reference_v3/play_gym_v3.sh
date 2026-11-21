#!/usr/bin/env bash
set -euo pipefail
amp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
amp_repo="${URDF4_REPO:-$(cd -- "$amp_dir/../../.." && pwd)}"
source "$amp_repo/unitree-rl/bin/activate"
amp_dataset="$amp_dir/gym_verified"
amp_commands="$amp_dir/commands"
amp_clip="forward_p0500_p0000_p0000"
amp_seconds=20
if [[ "${1:-}" == "--transitions" ]]; then
  shift
  amp_dataset="$amp_dir/gym_verified/transitions"
  amp_commands="$amp_dir/transition_commands"
  amp_clip="start_stop_forward"
  amp_seconds=30
fi
cd "$amp_repo"
export DISPLAY="${DISPLAY:-:1}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
exec python tools/urdf4_amp/play_isaac_reference.py \
  --dataset "$amp_dataset" --commands-dir "$amp_commands" \
  --closed-loop-teacher --direct-joint-targets --clip "$amp_clip" \
  --seconds "$amp_seconds" --follow --sim_device cuda:0 --rl_device cuda:0 "$@"
