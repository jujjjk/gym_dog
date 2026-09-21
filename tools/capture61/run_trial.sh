#!/usr/bin/env bash
# OPERATOR-ONLY: moving cases invoke the existing guarded motion client.
set -euo pipefail
case_name=${1:?Usage: run_trial.sh stand|forward|backward|left|right|turn-left|turn-right run01}
run=${2:?Provide run01, run02 or run03}
[[ "$run" =~ ^run[0-9]+$ ]] || { echo 'Invalid run label'; exit 2; }
session=$(cat /tmp/mydog_capture61_session)
source_csv="$session/cycles.csv"
if [[ "$case_name" == stand ]]; then
  exec ros2 run mydog_policy mydog_capture61_slice --source "$source_csv" --label "stand_$run" --seconds 30
fi
case "$case_name" in
  forward) speed=.15;;
  backward) speed=.10;;
  left|right) speed=.10;;
  turn-left|turn-right) speed=.20;;
  *) echo 'Unknown case'; exit 2;;
esac
[[ ! -e "$session/${case_name}_$run.csv" ]] || { echo 'Trial file already exists'; exit 2; }
ros2 run mydog_policy mydog_capture61_slice --source "$source_csv" --label "${case_name}_$run" --seconds 60 &
rec_pid=$!
trap 'kill -INT "$rec_pid" 2>/dev/null || true' INT TERM
sleep 1
if ! ros2 run mydog_policy mydog_model18000_sequence --action "$case_name" --speed "$speed" --seconds 30; then
  kill -INT "$rec_pid" 2>/dev/null || true
  wait "$rec_pid" || true
  echo 'Trial failed; inspect controller and saved data. No automatic retry.'
  exit 1
fi
wait "$rec_pid"
