"""Hardcoded diagonal-VMC teacher for RS01: env phase clock + Raibert placement + planar IK.

Outputs joint-target actions in the executed action space, so collected references pass
through the exact RS01 actuator/delay plant and match deployment semantics.
"""
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.dog.rs01_cpg import RS01FootTrajectory
LEGS = RS01FootTrajectory.LEGS


class Rs01MpcTeacher:
    # Front(+1)/rear(-1) split used for yaw steering, per LEGS order.
    X_SIGN = torch.tensor([1., 1., -1., -1.])
    GAINS = dict(height_d=.08, raibert_x=.5, raibert_y=.5,
                 pitch=.06, roll=.08, yaw_x=.10, lateral_lead=.5,
                 hip_max=.14, duty=.62, max_stride=.058, clearance=.030)
    # Stance foot reference = FK(default thigh/calf) relative to the hip:
    # (0, -0.30) m. The commanded joint band is only default+-0.22 rad, which
    # spans foot z in [-0.322, -0.257]; centering the stance at -0.30 leaves
    # symmetric swing (lift) and press headroom inside the reachable envelope.
    NOMINAL_Z = -0.300

    def __init__(self, env, gains=None):
        self.env=env
        self.traj=RS01FootTrajectory()
        self.g=dict(self.GAINS)
        if gains:self.g.update(gains)
        dev=env.device
        name_to_index={n:i for i,n in enumerate(env.dof_names)}
        self.joint_index={leg:{j:name_to_index[f"{leg}_{j}_joint"]
            for j in ("hip","thigh","calf")} for leg in LEGS}
        slots=[env.foot_slot_by_leg[leg] for leg in LEGS]
        self.foot_slot=torch.tensor(slots,dtype=torch.long,device=dev)
        self.default=env.default_dof_pos.reshape(-1).clone()
        self.scale=env.rs01_action_scale_rad.reshape(-1).clone()
        self.x_sign=self.X_SIGN.to(device=dev,dtype=torch.float)
        # Stance foot in the hip planar frame is the URDF FK of the default
        # joint pose, (0, -0.30) m; identical for all four legs.
        self.nominal_x=torch.zeros(4,device=dev)
        self.nominal_z=torch.full((4,),self.NOMINAL_Z,device=dev)
        stance_thigh,stance_calf=self.traj.inverse_kinematics(
            self.nominal_x,self.nominal_z)
        band_lo=self.default-self.scale
        band_hi=self.default+self.scale
        thigh_idx=torch.tensor([self.joint_index[l]["thigh"] for l in LEGS],device=dev)
        calf_idx=torch.tensor([self.joint_index[l]["calf"] for l in LEGS],device=dev)
        for name,val,idx in (("thigh",stance_thigh,thigh_idx),("calf",stance_calf,calf_idx)):
            lo,hi=float(band_lo[idx].min()),float(band_hi[idx].max())
            if float(val.min())<lo or float(val.max())>hi:
                raise RuntimeError(f"stance {name} IK {val.tolist()} outside "
                    f"reachable commanded band [{lo:.3f},{hi:.3f}]")

    def _leg_phases(self):
        phase=self.env._gait_phase()
        return torch.stack((phase,torch.remainder(phase+.5,1.),
            torch.remainder(phase+.5,1.),phase),1)

    def step_targets(self):
        env=self.env;g=self.g
        freq=(env.v25_frequency_hz if hasattr(env,"v25_frequency_hz")
            else torch.full((env.num_envs,),1./float(env.cfg.rewards.gait_period_s),
                device=env.device))
        period=1./freq
        cmd=env.commands[:,:3]
        vx,vy,wz=cmd[:,0],cmd[:,1],cmd[:,2]
        vel=env.base_lin_vel;rpy=env.rpy
        dx=((g["raibert_x"]*(vx-vel[:,0])*period+g["pitch"]*rpy[:,1])[:,None]
            +g["yaw_x"]*wz[:,None]*period[:,None]*self.x_sign[None,:]).clamp(-.04,.04)
        # Hip-referenced stance depth is fixed geometry, so no height term is
        # needed; only damp vertical body velocity for a stable stance.
        dz=g["height_d"]*env.root_states[:,9]
        leg_phase=self._leg_phases()
        duty=torch.full_like(leg_phase,g["duty"])
        # Stride capped to the RS01 action/rate envelope (default±0.22 rad, 2.6 rad/s).
        stride=(vx*period).clamp(-g["max_stride"],g["max_stride"])
        foot_x,foot_z=self.traj.sample(leg_phase,duty,stride.clone(),
            torch.full((env.num_envs,),g["clearance"],
                device=env.device),
            nominal_x_m=self.nominal_x[None,:]+dx,
            nominal_z_m=self.nominal_z[None,:]-dz[:,None])
        thigh,calf=self.traj.inverse_kinematics(
            foot_x.reshape(-1),foot_z.reshape(-1))
        # Keep saturated frames inside the workspace so actions stay finite.
        thigh=thigh.clamp(-1.6,.6);calf=calf.clamp(.15,1.9)
        thigh=thigh.reshape(env.num_envs,4);calf=calf.reshape(env.num_envs,4)
        # Lateral stepping: stance foot trails the body by vy*T/2, swing re-places it.
        lead=(g["lateral_lead"]*vy+g["raibert_y"]*(vy-vel[:,1]))*period \
            +g["roll"]*rpy[:,0]
        stance=leg_phase<duty
        progress=torch.where(stance,leg_phase/duty,
            (leg_phase-duty)/(1.-duty))
        y_off=torch.where(stance,lead[:,None]*(1.-progress),
            lead[:,None]*progress)
        hip=(y_off/self.nominal_z.abs()[None,:]).clamp(-g["hip_max"],g["hip_max"])
        q_target=self.default.expand(env.num_envs,-1).clone()
        for i,leg in enumerate(LEGS):
            idx=self.joint_index[leg]
            q_target[:,idx["hip"]]=hip[:,i]
            q_target[:,idx["thigh"]]=thigh[:,i]
            q_target[:,idx["calf"]]=calf[:,i]
        return q_target

    def actions(self):
        return (self.step_targets()-self.default)/self.scale
