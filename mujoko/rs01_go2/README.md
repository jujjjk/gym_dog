# RS01 new-machine Sim2Sim

This runner is independent of the Fanfan MuJoCo tooling. `Go2` in task names
describes the minimal reward/task layout; it is not a Unitree or previous-
generation physical model. Every newly exported policy is bound by SHA-256 to
`dog_urdf/urdf/dog_rs01.urdf`, the calibrated new-machine URDF.

Only schema-version-2 exports are accepted. Version 1 incorrectly routed actor
I/O in URDF declaration order (`FR, FL, RR, RL`), while Isaac Gym actually
exposes this asset as `FL, FR, RL, RR`. The default standing angles are equal
on all legs, so that error passed static standing checks but destroyed dynamic
diagonal walking. Re-export every checkpoint after this fix; old ONNX files
are intentionally rejected.

The runner preserves the 51-observation Kp40 policy contract and the measured
per-motor RS01 delay, first-order response, gain and Coulomb friction. The
identified actuator state advances every 5 ms. Contact is integrated in two
2.5 ms substeps to match the PhysX task, with explicit `solref`/`solimp`
calibrated against the new machine's zero-action standing equilibrium.

The actor output is used directly. There is no `tanh`, CPG offset, compensation
controller or ideal motor. The 17 N.m limit applies to electromagnetic motor
torque; friction is then applied separately without a second hidden clip.

Body-frame linear and angular velocity are computed from MuJoCo world-frame
object velocity using the floating-base quaternion. Do not use
`mj_objectVelocity(..., flg_local=1)` as the policy frame for this imported
model: its local-axis convention does not match Isaac Gym's
`quat_rotate_inverse`.

```bash
cd /home/nszb/gym

PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/export_policy.py \
  --task rs01_go2_straight_kp40 \
  unitree_rl_gym/logs/rs01_go2_straight_phase_load/Jul27_09-51-57_kp40_adapt_from635_pilot/model_730.pt \
  artifacts/rs01_go2_sim2sim/model_730.onnx

PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/prepare_model.py \
  artifacts/rs01_go2_sim2sim/model_730.onnx \
  artifacts/rs01_go2_sim2sim/dog_rs01_scene.xml

PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/sim2sim.py \
  --scene artifacts/rs01_go2_sim2sim/dog_rs01_scene.xml \
  --policy artifacts/rs01_go2_sim2sim/model_730.onnx \
  --duration 30 \
  --command 0.23 0 0 \
  --csv artifacts/rs01_go2_sim2sim/model_730_mujoco_30s.csv \
  --summary artifacts/rs01_go2_sim2sim/model_730_mujoco_30s.json
```

Add `--viewer` to the last command for real-time visualization.

For a checkpoint trained by the bridge task, select its exact contract during
export:

```bash
PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/export_policy.py \
  --task rs01_go2_sim2sim_adapt \
  <checkpoint.pt> \
  artifacts/rs01_go2_sim2sim/adapt_candidate.onnx
```

Always regenerate the scene after exporting a policy. New exports record the
exact URDF SHA-256, contact calibration, integration timestep and motor-step
timing. `sim2sim.py` refuses a policy/scene pair whose URDF hashes differ.

Current matched-model example:

```bash
cd /home/nszb/gym

PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/export_policy.py \
  --task rs01_go2_sim2sim_robust \
  unitree_rl_gym/logs/rs01_go2_straight_phase_load/\
Jul27_13-07-38_sim2sim_robust_from_kd050_840/model_870.pt \
  artifacts/rs01_go2_sim2sim/model_870_matched.onnx

PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/prepare_model.py \
  artifacts/rs01_go2_sim2sim/model_870_matched.onnx \
  artifacts/rs01_go2_sim2sim/dog_rs01_scene_matched.xml

PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
/home/nszb/gym/unitree-rl/bin/python \
mujoko/rs01_go2/sim2sim.py \
  --scene artifacts/rs01_go2_sim2sim/dog_rs01_scene_matched.xml \
  --policy artifacts/rs01_go2_sim2sim/model_870_matched.onnx \
  --duration 30 \
  --command 0.23 0 0 \
  --csv artifacts/rs01_go2_sim2sim/model_870_matched_mujoco_30s.csv \
  --summary artifacts/rs01_go2_sim2sim/model_870_matched_mujoco_30s.json
```
