# B18000 lateral drift feedback candidate

Superseded: the operator reported straight travel with the IMU balanced and
requested cancellation of extra compensation. The current local implementation
removes both this lateral feedback and the fixed right-yaw bias. It retains the
existing symmetric heading feedback, 2-degree deadband and straight-command
planar mapping. It does not restore the earlier heading-to-lateral rotation.
Safety limits and the trained hip target mapping remain unchanged.

Replacement Jetson workspace (built, not started):
`/home/jetson/deploy_staging/b18000-no-extra-compensation-20260920`.
Use its `install/setup.bash` in each new terminal on the next manual restart;
an already-running controller is not changed by this deployment.
Status reports `extra_compensation_enabled=false`,
`lateral_feedback_valid=false`, and `lateral_correction_mps=0`.

The following describes the retired experiment, not the current implementation.

2026-09-20: body moves right with its heading approximately unchanged.
This change adds symmetric body-y velocity feedback for straight translation.
It does not change mechanical zeros, torque protection, existing heading trim,
or the original operator command in the observation. The actor receives the
corrected lateral command and coordinates all four legs.

Correction uses gain 0.5, a 0.005 m/s deadband, a 0.3 s low-pass filter,
a 0.02 m/s cap, and a 0.02 m/s per second slew limit. It requires 0.3 s
of continuous valid feedback, healthy/ready heading consistency, legal inferred
diagonal support, and both guard and actor odometry confidence >= 0.6.
Stand, march, explicit lateral/turn commands and invalid feedback clear the trim.
Inferred contact is not independent proof of ground contact; suspended motion
is not a valid test of ground translation. No position is integrated from odometry.

Status adds `lateral_feedback_valid` (external gate, not proof correction is active)
and `lateral_correction_mps`. These do not replace existing observer warnings.

Latest 18:25:40 log: 58/496 walk samples met the external and logged odometry
gates, with only 0.12 s maximum continuous eligibility. This candidate would
therefore not engage on that record. Resolve feedback validity before assessing
its physical effectiveness; do not reduce thresholds merely to make it engage.
No physical correction or stability has been validated.

Jetson 172.19.18.130 isolated built workspace:
`/home/jetson/deploy_staging/b18000-lateral-candidate-20260920`.
28 offline tests passed there, including mocked ROS node/startup tests and
lateral sign, saturation, reset, slew and observation-routing regressions.
The candidate was built but not launched. No motor commands were sent.
