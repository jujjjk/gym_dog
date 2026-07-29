# model_1725 54-D Sim2Sim baseline

- Source task: `rs01_go2_model1425_path54`
- Checkpoint: `Jul29_10-13-17_path54_from1425_023/model_1725.pt`
- Command: `0.23 m/s`
- Policy/control rate: `50 Hz`
- MuJoCo motor step: `5 ms`
- MuJoCo integration: two `2.5 ms` contact substeps
- URDF SHA-256:
  `48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4`

## 30-second MuJoCo result

| Metric | Result |
|---|---:|
| Mean forward speed | 0.2288 m/s |
| Lateral path RMS | 0.2095 m |
| Final lateral displacement | 0.2365 m |
| Unwrapped yaw drift | 0.0709 rad |
| Yaw-rate RMS | 0.2587 rad/s |
| Roll RMS | 0.0672 rad |
| Exact desired contact | 75.93% |
| Flight | 0% |
| Fall | no |
| Raw torque P95 | 17.78 N.m |
| Raw torque maximum | 26.06 N.m |
| Peak saturation | 6.57% |
| Motor above 6 N.m | 30.69% |
| Illegal ground contact | 0 frames |

## 60-second MuJoCo result

| Metric | Result |
|---|---:|
| Mean forward speed | 0.2279 m/s |
| Lateral path RMS | 0.2606 m |
| Final lateral displacement | 0.4335 m |
| Unwrapped yaw drift | 0.1511 rad |
| Yaw-rate RMS | 0.2547 rad/s |
| Roll RMS | 0.0666 rad |
| Exact desired contact | 76.03% |
| Flight | 0% |
| Fall | no |
| Raw torque P95 | 17.79 N.m |
| Peak saturation | 6.63% |

The 54-D observation and actuator runtime contract pass, and the path feedback
reduces the old model_930 30-second final lateral displacement from about
1.287 m to 0.237 m. The residual 60-second drift and higher MuJoCo motor usage
show that the nominal model_1725 is still sensitive to contact/actuator
differences. Use `rs01_go2_path54_sim2sim_transfer` for the next short
checkpoint-selection stage; do not alter the accepted gait or motor contract.
