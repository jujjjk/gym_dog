#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../src/mydog_policy"
export PYTHONPATH="$PWD:/usr/local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
python3 -m pytest -q test/test_b23500_ready_switch.py test/test_heading_reference.py test/test_b23500_transition.py test/test_odometry_quality.py test/test_observation_pipeline.py test/test_b23500_straight_only_correction.py test/test_rs01_model23500.py test/test_rs01_model23500_fast.py test/test_rs01_model6850_node_offline.py test/test_b23500_continuous.py test/test_rt_schedule.py test/test_common_time_alignment.py test/test_b23500_telemetry_budget.py test/test_rs01_walk_inhibit.py test/test_rs01_model6850_guard.py test/test_capture61.py test/test_capture61_node.py
