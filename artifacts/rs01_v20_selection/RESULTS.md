# V20 selection

## Recommendation

Recommend **B/model_18000.pt for reduced adduction and front-foot bias**, among the eight
screened candidates. This is not an exhaustive optimum or a real-robot acceptance claim.
A/model_17000.pt has slightly better tracking and contact matching; B wins the user's
current geometric priorities, not every metric.

B run: `/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v20_bounded_hip/Sep13_18-35-41_v20_gpu1_seed16_20260913_183537`.
SHA256 model18000: `0d6c90c84742c53d679cefa73fb3f409d5254b0fba8280313f59bb0fd71c8b57`.
A run: `/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v20_geometry/Sep13_18-35-40_v20_gpu0_seed16_20260913_183537`.
Both trained seed16, iterations15000–18000. No training or behavior edits in this evaluation.

## Repository / physical contract

Branch `fix/rs01-heading-odom-soft-inhibit`, HEAD `3a0a7f4`, existing dirty state preserved.
Actual RS01 URDF SHA256 `48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4`.
Previous audited geometry unchanged: mass11.7317kg, thigh180mm/calf202.158mm, point-foot radius16mm.
Identified RS01 actuator, PD, rate/acceleration, delay, guard and thermal behavior unchanged.
B uses its conditional hip mapping before the limiter; ordinary V19 deployment is incompatible.

## Protocol

- Screen each run at16000/17000/17500/18000:24command cases×4env×10s, eval seed20260923;
  23march/motion cases used for aggregate ranking, static stand separate.
- Each candidate posture: march, forward .2m/s, backward -.2m/s;4env×12s,
  eval seed20260918, measurement6–12s.
- Finalists A17000/B18000: same wide suite×4env×60s, eval seed20260924,
  measurement2–60s. Final posture30s, measure6–30s, eval seed20260918.
- Nominal flat terrain, noise/randomization/pushes off. Policy50Hz, real physics dt unchanged;
  only wallclock pacing skipped. No exhaustive checkpoint sweep,3–5minute thermal validation,
  randomized robustness, MuJoCo, ONNX or real tests.

## Screen

| Group/model | Resets | Max observed flight ratio | Mean roll RMS deg | Mean vz RMS m/s | Raw PD peak Nm |
|---|---:|---:|---:|---:|---:|
| A16000 | 1 | .0006 | 1.73 | .086 | 26.19 |
| A17000 | 0 | 0 | 1.48 | .086 | 15.66 |
| A17500 | 0 | 0 | 1.59 | .088 | 15.17 |
| A18000 | 0 | 0 | 1.71 | .086 | 20.41 |
| B16000 | 0 | 0 | 1.68 | .089 | 16.02 |
| B17000 | 0 | 0 | 1.61 | .084 | 15.98 |
| B17500 | 0 | 0 | 1.61 | .084 | 15.60 |
| B18000 | 0 | 0 | 1.57 | .086 | 15.03 |

A16000 rejected. A17000 selected for lower roll/better overall balance of bias and torque.
B18000 selected for small march/forward bias, wide feet and modest raw torque;
B17000 had smaller reverse bias in screening but narrower support.

## Final60s wide results

Case averages are unweighted over23motion cases. Peaks are maxima; meanP95 is not pooledP95.

| Metric | A17000 | B18000 |
|---|---:|---:|
| Resets / observed flight50Hz | 0 / 0 | 0 / 0 |
| Roll RMS deg | 1.52 | 1.57 |
| Vertical velocity RMS m/s | .083 | .083 |
| vx/vy/wz RMSE m/s,m/s,rad/s | .0252/.0214/.0707 | .0284/.0218/.0816 |
| Four-foot contact | 22.87% | 21.67% |
| Exact desired contact match | 72.62% | 71.27% |
| Raw PD peak Nm | 16.46 | 16.76 |
| Mean case raw PD P95 Nm | 6.77 | 6.78 |
| Peak joint speed rad/s | 17.44 | 17.92 |

Raw PD is not applied torque. Fixed14Nm saturation averages approximately .001%/.016%,
not dynamic active-limit saturation. Finite observations/rewards for both.

## Final30s posture results

Positive RF lead means right-front foot ahead of left in body frame. Negative means left ahead.
These are mean foot positions, not identical-frame mirroring requirements.

| Command | Old V19 RF lead mm | A RF lead mm | B RF lead mm | A front width mm | B front width mm |
|---|---:|---:|---:|---:|---:|
| March | 51.1 | 8.3 | 10.6 | 213.0 | 252.3 |
| Forward .2 | 20.3 | -2.0 | -3.7 | 213.3 | 264.5 |
| Reverse -.2 | 63.3 | 41.2 | 25.4 | 199.3 | 261.3 |

Old V19 is seed16/model15000, same posture seed/duration/window as final30.
B inward hip means FL/FR/RL/RR, degrees:

- March:4.0/4.2/2.4/2.1.
- Forward:3.6/2.2/1.3/1.2.
- Reverse:2.4/3.9/3.0/2.1.

No observed pace-only support or single-foot support in these three posture tests.
Point-foot 'inward toe' is hip/lower-leg adduction, not independent toe yaw.

## B measured commands

| Command vx/vy/wz | Measured mean vx/vy/wz |
|---|---|
| March0/0/0 | .005/.001/.001 |
| Forward .2/0/0 | .204/.005/.003 |
| Reverse -.2/0/0 | -.208/.005/-.000 |
| Left0/.2/0 | -.015/.204/-.003 |
| Right0/-.2/0 | .031/-.203/.000 |
| Yaw0/0/.3 | .003/.008/.283 |
| Yaw0/0/-.3 | .012/.005/-.354 |
| Combined .3/.1/.25 | .321/.097/.237 |
| Fast forward .6/0/0 | .516/-.002/.003 |

Remaining shortcomings: reverse bias25mm, lateral-command longitudinal drift,
yaw-right overshoot, high-speed undertracking, persistent bounce. Nominal passing is not
proof of real tipping resistance. Wider feet and lower adduction are measured improvements,
but no real COM/support-margin causality or untethered readiness is established.

## Reproduce / rollback

Wide helper: `python legged_gym/scripts/rs01_eval_fast.py wide --task=TASK --load_run=RUN --checkpoint=CKPT --duration_s=60 --eval_envs=4 --eval_suite=wide --seed=20260924 --headless --sim_device=cuda:0 --rl_device=cuda:0`.
Posture: `RS01_AUDIT_OUTPUT_DIR=OUTPUT python /home/nszb/gym/artifacts/rs01_v18_selection/posture_audit.py --task=TASK --load_run=RUN --checkpoint=CKPT --duration_s=30 --eval_envs=4 --seed=20260918 --headless --sim_device=cuda:0 --rl_device=cuda:0`.
Logs/JSON/NPZ retained in A/ and B/ (final posture in final30/).
No behavior changed: rollback simply selects old task/checkpoint. All prior models retained.

## Final switching test

Both finalists completed25×5s=125s,4env,eval seed20260924,0resets and finite outputs.
March is interleaved with12motion commands; no random stand.
JSONs under `artifacts/rs01_v16_selection/`:

- A: `sequence_17000_20260924_1789302646483675561.json`.
- B: `sequence_18000_20260924_1789302671150803166.json`.

### B viewer command (5 seconds per action)

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python /home/nszb/gym/artifacts/rs01_v16_selection/play_sequence.py \
  --task=rs01_omni_v20_bounded_hip \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v20_bounded_hip/Sep13_18-35-41_v20_gpu1_seed16_20260913_183537 \
  --checkpoint=18000 --num_envs=1 --seed=20260924 \
  --sim_device=cuda:0 --rl_device=cuda:0
```
