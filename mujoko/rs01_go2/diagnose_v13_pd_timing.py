"""Diagnostic A/B only; never changes production training or legacy runner."""
import json
from pathlib import Path
import numpy as np
import mujoco
from sim2sim_v13 import V13Sim,roll_pitch_yaw
from sim2sim import compute_rs01_torques


class SubstepPD(V13Sim):
    def physics_step(self):
        # Keep identified response, delay and motor target update at 5 ms.
        self.target_history=np.roll(self.target_history,1,axis=0)
        self.target_history[0]=self.limited_target
        delayed=self.target_history[self.delay_steps,np.arange(12)]
        self.response_target+=self.response_alpha*(
            self.default+self.response_gain*(delayed-self.default)-self.response_target)
        for _ in range(self.integration_substeps):
            raw,motor,applied=compute_rs01_torques(
                self.response_target,self.data.qpos[self.qpos_indices],
                self.data.qvel[self.qvel_indices],self.kp,self.kd,self.peak_limit,
                self.friction,self.friction_smoothing)
            self.data.ctrl[self.actuator_indices]=applied
            mujoco.mj_step(self.model,self.data)
            self.last_raw_torque=raw;self.last_motor_torque=motor;self.last_applied_torque=applied


if __name__=='__main__':
    out=Path(__file__).resolve().parents[2]/'artifacts/rs01_v13_a6850_sim2sim'
    results=[]
    for mode,cls in [('original_5ms',V13Sim),('diagnostic_2p5ms',SubstepPD)]:
        for vx in (.2,.4):
            for phase in (0.,.25,.5,.75):
                sim=cls(out/'scene.xml',out/'model_6850.onnx',[vx,0,0]);sim.phase_value=phase
                rows=[];reason=None
                for i in range(1500):
                    sim.control_step();r,p,y=roll_pitch_yaw(sim.data.qpos[3:7])
                    rows.append([*sim.base_velocity_body()[0][:2],sim.base_velocity_body()[1][2],
                                 *sim.last_raw_torque,*sim.last_motor_torque])
                    if abs(r)>.8 or abs(p)>.8 or sim.data.qpos[2]<.18:reason='fall';break
                x=np.array(rows);steady=x[100:]
                result=dict(mode=mode,vx=vx,phase=phase,seconds=(i+1)*.02,stop=reason,
                    mean_velocity=steady[:,:3].mean(0).tolist() if len(steady) else None,
                    raw_p95_first_0p2_to_1s=float(np.percentile(abs(x[9:50,3:15]),95)),
                    raw_p95_full=float(np.percentile(abs(x[:,3:15]),95)),
                    motor_peak=float(abs(x[:,15:]).max()),
                    finite=bool(np.isfinite(x).all()))
                results.append(result);print(json.dumps(result),flush=True)
    (out/'pd_timing_ab.json').write_text(json.dumps(results,indent=2)+'\n')
