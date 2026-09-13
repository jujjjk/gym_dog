# B18000 MuJoCo bridge and smoke

Selected source: V20 bounded-hip run `Sep13_18-35-41_v20_gpu1_seed16_20260913_183537/model_18000.pt`.
Checkpoint SHA256: `0d6c90c84742c53d679cefa73fb3f409d5254b0fba8280313f59bb0fd71c8b57`.
Actual RS01 URDF SHA256: `48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4`.
11.7317kg,180mm thigh/202.158mm calf; regenerated scene from exported contract and actual URDF.
Existing dirty working tree preserved; no training or real deployment changes.

## Missing bridge features addressed

New `export_v20.py` exports61→12 actor and explicit mapping/guard metadata.
New `sim2sim_v20.py` reuses V14 identified RS01 plant and fresh sensor timestamps,
legacy sensor-only odometry and frozen stand phase. It adds the exact V20 conditional inward
target mapping before rate/acceleration limiting and V16 policy-boundary torque projection,
with next-tick guard derating and heating retained across resets.
Limited actor target history is not overwritten by guarded sent targets.
PD2.5ms, identified response5ms, policy20ms, delay/FOPDT/friction and14Nm operating cap retained.
No ideal motors, q/dq state projection, automatic resets, or emergency recovery.
Old bridges unchanged. Bare B ONNX remains incompatible with old real deployment code.

## Checks

- ONNX versus PyTorch output max absolute difference5.722e-6 (8random61D probes).
- `python mujoko/rs01_go2/check_v20_mapping.py`:1000 randomized mapping/guard/thermal
  algebra comparisons against training torch implementation, absolute tolerance1e-12 passed.
  This is algebra parity, not a full cross-engine state-trajectory equivalence proof.
- Same25segment sequence as PhysX: march interleaved with12movements,5s each, total125s.
  No random stand.4initial phases tested, not4randomized terrain seeds.

| Initial phase | Completed s | Stop reason | Observed flight50Hz | Roll RMS deg | vz RMS m/s |
|---|---:|---|---:|---:|---:|
| 0 | 125 | none | 0 | 1.53 | .105 |
| .25 | 125 | none | 0 | 1.57 | .105 |
| .5 | 125 | none | 0 | 1.58 | .105 |
| .75 | 125 | none | 0 | 1.59 | .105 |

All finite. Phase0 sampled raw PD peak19.31Nm (NOT applied torque), max joint speed18.21rad/s.
CSV/JSON `sequence_p*.csv/.json` retain outcomes; rollout stops rather than resets on
height<.18m, abs roll/pitch>.8rad, nonfinite state, or physical overspeed.
Vertical oscillation remains. Do not directly compare this whole-sequence .105m/s to the
PhysX .083m/s figure averaged over fixed-command60s cases: protocols differ.
This is a nominal Sim2Sim demonstration, not thermal certification or Sim2Real readiness.
Viewer flag is provided; GUI itself was not launched during headless validation.

## Demonstration

```bash
cd /home/nszb/gym
source /home/nszb/gym/unitree-rl/bin/activate
python mujoko/rs01_go2/sim2sim_v20.py \
  --policy artifacts/rs01_v20_sim2sim/B18000.onnx \
  --scene artifacts/rs01_v20_sim2sim/scene.xml \
  --sequence --viewer \
  --output artifacts/rs01_v20_sim2sim/viewer_B18000
```

Rollback: choose old policy with its matching old bridge; all old exports/checkpoints retained.
