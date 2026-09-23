"""V22 MuJoCo: shared Torch sensor/odometry/61D assembly, real RS01 plant."""
import sys
import math
import json
from pathlib import Path
from types import SimpleNamespace
import isaacgym  # before torch
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'unitree_rl_gym'))
from legged_gym.envs.rs01_omni_v2.rs01_sensor_snapshot import SensorSnapshot, assemble61
from legged_gym.envs.rs01_go2_straight.rs01_odometry import Rs01TorchLegOdometry
import sim2sim_v20 as bridge
from sim2sim import limited_target_step, wrapped_angle


def tensor(a):return torch.as_tensor(np.asarray(a).copy(),dtype=torch.float32).reshape(1,-1)


class V22Sim(bridge.V20Sim):
    task_name='rs01_omni_v22_sensor'
    sensor_profile='normal'
    seed=20260925

    def __init__(self,*args):
        super().__init__(*args)
        cfg=self.cfg['v22']
        if cfg['host_guard']!='policy_sensor_snapshot':raise ValueError('Wrong V22 guard contract')
        torch.set_num_threads(1);torch.manual_seed(self.seed)
        sc=SimpleNamespace(**cfg['sensor'])
        sc.enabled=self.sensor_profile!='clean'
        sc.clean_fraction=.25 if self.sensor_profile=='mixed' else 0.
        sc.robust_fraction={'clean':0.,'normal':0.,'robust':1.,'mixed':.25}[self.sensor_profile]
        self.sensor=SensorSnapshot(1,'cpu',self.physics_dt,sc)
        od=cfg['odometry']
        self.sensor_odometry=Rs01TorchLegOdometry(1,'cpu',nominal_base_height=od['nominal_base_height_m'],
            foot_radius=od['foot_radius_m'],height_margin=od['height_margin_m'],
            vertical_speed_threshold=od['vertical_speed_threshold_m_s'],
            velocity_residual_threshold=od['velocity_residual_threshold_m_s'],filter_alpha=od['filter_alpha'],
            no_contact_decay=od['no_contact_decay'],previous_stance_score_bonus=od['previous_stance_score_bonus'],
            strict_diagonal_pairs=od.get('strict_diagonal_pairs',False))
        self.sensor.reset(torch.tensor([0]),0,*self.sensor_truth());self.sensor.read(0)
        self.heading_target=float(self.sensor.yaw[0]);self.sensor_captured=-1
        self.update_estimator()

    def sensor_truth(self):
        # qpos/qvel after mj_step are fresh; cvel can still be pre-integration.
        _,angular=self.base_velocity_body()
        q=self.data.qpos[3:7]  # MuJoCo wxyz -> training xyzw
        return (tensor(self.data.qpos[self.qpos_indices]),tensor(self.data.qvel[self.qvel_indices]),
                tensor(q[[1,2,3,0]]),tensor(angular),tensor(self.last_motor_torque))

    def capture(self):
        if self.sensor_captured!=self.feedback_tick:
            self.sensor.acquire(self.feedback_tick,*self.sensor_truth())
            self.sensor_captured=self.feedback_tick

    def update_estimator(self):
        self.estimate=self.sensor_odometry.estimate(self.sensor.q,self.sensor.dq,self.sensor.gyro)
        self.last_estimated_linear_velocity=self.estimate['base_linear_velocity'][0].numpy().copy()
        self.confidence=float(self.estimate['confidence'][0])

    def set_command(self,command,gait=1.):
        old=self.turning
        super().set_command(command,gait)
        if old!=self.turning:self.heading_target=float(self.sensor.yaw[0])

    def effective_command(self):
        c=self.cfg['v13']['commands']
        error=wrapped_angle(self.heading_target-float(self.sensor.yaw[0]))
        blend=np.clip(1-abs(self.command[2])/c['direction_blend_yaw_rad_s'],0,1)*(not self.turning)
        angle=np.clip(error,-c['direction_rotation_limit_rad'],c['direction_rotation_limit_rad'])
        rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        target=self.command.copy();correction=(rot@target[:2]-target[:2])*blend
        correction*=min(1,c['direction_planar_correction_limit_m_s']/max(np.linalg.norm(correction),1e-6))
        target[:2]+=correction;limit=c['direction_yaw_correction_limit_rad_s']
        target[2]+=blend*limit*math.tanh(c['direction_heading_gain']*error/limit)
        for i,key in enumerate(('lin_vel_x','lin_vel_y','ang_vel_yaw')):
            target[i]=np.clip(target[i],*self.cfg['commands']['ranges'][key])
        return target

    def observation(self):
        o=self.obs_cfg
        scales=SimpleNamespace(lin_vel=o['lin_vel_scale'],ang_vel=o['ang_vel_scale'],
                               dof_pos=o['dof_pos_scale'],dof_vel=o['dof_vel_scale'])
        self.last_observation=assemble61(self.sensor,self.estimate['base_linear_velocity'],
            self.estimate['confidence'],tensor(self.effective_command()),tensor(self.command),tensor(self.default),
            scales,tensor(o['command_scale']),tensor(self.action),torch.tensor([self.phase],dtype=torch.float32),
            torch.tensor([wrapped_angle(self.heading_target-float(self.sensor.yaw[0]))],dtype=torch.float32),
            torch.tensor([self.gait_enable],dtype=torch.float32)).clamp(-o['clip'],o['clip'])[0].numpy()
        return self.last_observation

    def update_policy(self):
        action=self.session.run(['actions'],{'observations':self.observation()[None,:]})[0][0]
        self.action=np.clip(action.astype(np.float64),-self.action_clip,self.action_clip)
        desired=bridge.map_targets(self.default+self.action_scale*self.action,self.command,
            self.hip_ids,self.hip_sides,self.action_scale[self.hip_ids]*self.action_clip,self.cfg['v20'])
        self.desired_target=desired.copy()
        self.limited_target,self.target_rate=limited_target_step(desired,self.limited_target,self.target_rate,
            self.rate_limit,self.acceleration_limit,self.policy_dt,reached_atol=1e-12)
        self.guard_target,self.guard_raw_pd,self.guard_safe_pd=bridge.project_guard(self.limited_target,
            self.sensor.q[0].numpy(),self.sensor.dq[0].numpy(),self.kp,self.kd,self.guard_limit,self.lower,self.upper)
        self.guard_rms_sq,self.guard_limit=bridge.update_guard(self.guard_rms_sq,
            np.maximum(abs(self.guard_safe_pd),abs(self.sensor.torque[0].numpy())),self.policy_dt,self.cfg['v20']['guard'])

    def control_step(self):
        self.step_overspeed=False;self.step_max_speed=0.
        self.update_policy()
        for _ in range(self.decimation):
            self.capture();self.physics_step()
        self.capture();self.sensor.read(self.feedback_tick)
        self.policy_steps+=1
        self.phase_value=(self.phase_value+self.policy_dt*self.frequency())%1
        self.heading_target=wrapped_angle(self.heading_target+self.policy_dt*self.command[2])
        self.update_estimator()


FAST_MOVEMENTS=[('forward',(.4,0,0)),('backward',(-.3,0,0)),('left',(0,.3,0)),('right',(0,-.3,0)),
    ('forward_left',(.3,.15,0)),('forward_right',(.3,-.15,0)),('backward_left',(-.3,.15,0)),
    ('backward_right',(-.3,-.15,0)),('turn_left',(0,0,.6)),('turn_right',(0,0,-.6)),
    ('combined',(.3,.15,.5)),('combined_reverse',(-.3,-.15,-.5))]


if __name__=='__main__':
    p=bridge.get_parser();p.add_argument('--sequence',action='store_true');p.add_argument('--fast',action='store_true')
    p.add_argument('--sensor_profile',choices=('clean','normal','robust','mixed'),default='normal')
    p.add_argument('--sensor_seed',type=int,default=20260925)
    args=p.parse_args()
    if not args.sequence:p.error('V22 currently supports --sequence only')
    if not 0<=args.phase<1:p.error('phase must be in [0,1)')
    V22Sim.sensor_profile=args.sensor_profile;V22Sim.seed=args.sensor_seed
    bridge.V20Sim=V22Sim
    if args.fast:bridge.MOVEMENTS=FAST_MOVEMENTS
    with torch.no_grad():bridge.sequence_run(args)
    report=Path(str(args.output)+'.json')
    result=json.loads(report.read_text())
    result.update(sensor_profile=args.sensor_profile,sensor_seed=args.sensor_seed,fast=args.fast,
                  sensor_model='shared_training_SensorSnapshot',host_guard='policy_sensor_snapshot')
    report.write_text(json.dumps(result,indent=2)+'\n')
