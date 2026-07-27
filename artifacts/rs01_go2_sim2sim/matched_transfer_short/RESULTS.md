# RS01 matched-scene short-transfer result

- Source: `Jul27_13-07-38_sim2sim_robust_from_kd050_840/model_870.pt`
- Task: `rs01_go2_sim2sim_matched_transfer`
- Run: `Jul27_14-22-47_matched_mujoco_transfer_short_from870`
- Updates: 30, checkpoints every 5
- Command: 0.23 m/s
- Hard acceptance: regenerated new-machine matched MuJoCo scene, 30 seconds

## PhysX 10-second checkpoint screen

| model | vx m/s | final abs lateral m | yaw-rate RMS rad/s | exact contact % | flight % | raw P95 Nm | raw >17 % | resets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 870 | 0.224 | 0.052 | 0.279 | 67.6 | 0.0 | 16.05 | 3.45 | 0 |
| 875 | 0.225 | 0.024 | 0.272 | 67.7 | 0.0 | 16.05 | 3.38 | 0 |
| 880 | 0.227 | 0.033 | 0.253 | 67.9 | 0.0 | 15.95 | 3.28 | 0 |
| 885 | 0.225 | 0.036 | 0.255 | 68.2 | 0.0 | 15.98 | 3.42 | 0 |
| 890 | 0.229 | 0.021 | 0.243 | 68.4 | 0.0 | 15.94 | 3.17 | 0 |
| 895 | 0.229 | 0.031 | 0.244 | 68.6 | 0.0 | 15.97 | 3.23 | 0 |
| 900 | 0.228 | 0.042 | 0.240 | 69.8 | 0.0 | 16.08 | 3.45 | 0 |

## Matched MuJoCo 30-second hard acceptance

| model | completed 30 s | body vx m/s | yaw-rate RMS rad/s | exact contact % | flight % | raw P95 Nm | saturation % |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 870 | no, fell 8.04 s | 0.008 | 0.632 | 6.72 | 0.50 | 21.10 | 11.75 |
| 875 | yes | -0.001 | 0.534 | 8.13 | 0.07 | 19.81 | 9.38 |
| 880 | yes | -0.001 | 0.530 | 8.53 | 0.13 | 19.34 | 8.37 |
| 885 | yes | -0.001 | 0.636 | 6.73 | 1.53 | 20.44 | 11.29 |
| 890 | yes | -0.001 | 0.527 | 11.40 | 0.07 | 19.56 | 8.61 |
| 895 | no, fell 3.66 s | 0.022 | 0.496 | 10.38 | 0.55 | 22.13 | 13.66 |
| 900 | yes | -0.001 | 0.501 | 11.53 | 0.27 | 19.08 | 7.96 |

`model_900` is the best retained PhysX candidate and the best torque/contact
tradeoff in the matched scene. It is **not accepted for deployment** because
matched-MuJoCo forward body velocity is effectively zero and unwrapped yaw
reaches about -3.29 rad. More iterations with the same setup are not justified;
the remaining discrepancy must be isolated at the dynamic interface/contact
level before another migration run.
