"""Generic Fanfan MuJoCo runner driven entirely by ONNX deployment metadata."""
from pathlib import Path
import argparse,json,time,mujoco,numpy as np,onnxruntime as ort

def load_contract(session):
    meta=session.get_modelmeta().custom_metadata_map
    if "fanfan_deployment_config" not in meta: raise RuntimeError("ONNX lacks fanfan_deployment_config metadata; re-export it")
    c=json.loads(meta["fanfan_deployment_config"])
    if c.get("schema_version")!=1: raise RuntimeError(f"Unsupported config schema: {c.get('schema_version')}")
    return c

def gravity(q):
    r=np.empty(9);mujoco.mju_quat2Mat(r,q);return r.reshape(3,3).T@np.array([0,0,-1.])

class Sim:
    def __init__(self,model,policy,command=None):
        self.m=mujoco.MjModel.from_xml_path(str(model));self.d=mujoco.MjData(self.m);self.net=ort.InferenceSession(str(policy));self.cfg=load_contract(self.net)
        c=self.cfg;ctl=c["control"];obs=c["observations"];gait=c["gait"]
        self.names=c["joint_names"];self.default=np.asarray(c["default_joint_angles"],np.float32);self.kp=np.asarray(ctl["stiffness"],np.float32);self.kd=np.asarray(ctl["damping"],np.float32);self.scale=np.asarray(ctl["action_scale"],np.float32);self.limits=np.asarray(ctl["torque_limits"],np.float32)
        self.command=np.asarray(command if command is not None else c["commands"]["default"],np.float32);self.obs_cfg=obs;self.gait=gait
        if abs(self.m.opt.timestep-ctl["sim_dt"])>1e-9: raise RuntimeError(f"MJCF timestep {self.m.opt.timestep} != exported cfg {ctl['sim_dt']}")
        self.decimation=int(ctl["decimation"]);self.q=np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in self.names]);self.v=np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in self.names]);self.aid=np.array([mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,x+"_motor") for x in self.names]);self.bid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,"Trunk");self.reset()
    def reset(self):
        mujoco.mj_resetData(self.m,self.d);pos=self.cfg["initial_state"]["base_position"];xyzw=self.cfg["initial_state"]["base_quaternion_xyzw"];self.d.qpos[:7]=[*pos,xyzw[3],xyzw[0],xyzw[1],xyzw[2]];self.d.qpos[self.q]=self.default;self.action=np.zeros(len(self.names),np.float32);self.target=self.default.copy();self.n=0;mujoco.mj_forward(self.m,self.d)
    def gait_offset(self,phase):
        out=np.zeros(len(self.names),np.float32);r=self.gait["stance_ratio"]
        for i,name in enumerate(self.names):
            leg=name[:2];p=(phase+self.gait["phase_offsets"][leg])%1;s=np.clip((p-r)/(1-r),0,1);smooth=s*s*(3-2*s)
            if "thigh" in name:out[i]=self.gait["thigh_amplitude"]*(-1+2*np.clip(p/r,0,1) if p<r else 1-2*smooth)
            elif "calf" in name:out[i]=self.gait["calf_amplitude"]*np.sin(np.pi*smooth)*(p>=r)
        return out
    def policy(self):
        vel=np.empty(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.bid,vel,1);phase=(self.n*self.m.opt.timestep*self.decimation%self.gait["period"])/self.gait["period"];o=self.obs_cfg
        obs=np.concatenate((vel[3:]*o["lin_vel_scale"],vel[:3]*o["ang_vel_scale"],gravity(self.d.qpos[3:7]),self.command*np.asarray(o["command_scale"]),(self.d.qpos[self.q]-self.default)*o["dof_pos_scale"],self.d.qvel[self.v]*o["dof_vel_scale"],self.action,[np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)])).astype(np.float32)
        if obs.size!=self.cfg["dimensions"]["observations"]:raise RuntimeError(f"Observation size {obs.size} != export {self.cfg['dimensions']['observations']}")
        raw=self.net.run(["raw_actions"],{"observations":np.clip(obs,-o["clip"],o["clip"])[None]})[0][0];self.action=np.tanh(raw) if self.cfg["control"]["output_transform"]=="tanh" else raw;self.target=self.default+self.scale*self.action+self.gait_offset(phase);self.n+=1
    def step(self):
        raw=self.kp*(self.target-self.d.qpos[self.q])-self.kd*self.d.qvel[self.v];self.d.ctrl[self.aid]=np.clip(raw,-self.limits,self.limits);mujoco.mj_step(self.m,self.d);return raw

def main():
    root=Path(__file__).parent;p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=root/"models/fanfan_scene.xml");p.add_argument("--policy",type=Path,default=root/"models/fanfan_best.onnx");p.add_argument("--duration",type=float,default=20);p.add_argument("--command",nargs=3,type=float);p.add_argument("--viewer",action="store_true");a=p.parse_args();s=Sim(a.model,a.policy,a.command)
    if a.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(s.m,s.d) as v:
            v.cam.distance=1.4;v.cam.azimuth=135;v.cam.elevation=-18;t=time.time();ep=t;i=0
            while v.is_running() and time.time()-t<a.duration:
                st=time.time()
                if st-ep>=s.cfg["episode_length_s"]:s.reset();ep=st;i=0
                if i%s.decimation==0:s.policy()
                s.step();v.cam.lookat[:]=s.d.qpos[:3];v.sync();i+=1;time.sleep(max(0,s.m.opt.timestep-(time.time()-st)))
    else:
        raw=[];start=s.d.qpos[:2].copy();minz=99
        for i in range(int(a.duration/s.m.opt.timestep)):
            if i%s.decimation==0:s.policy()
            raw.append(s.step());minz=min(minz,s.d.qpos[2])
        x=np.abs(raw);q=s.d.qpos[3:7];yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));d=s.d.qpos[:2]-start
        print(f"duration_s={a.duration:.3f}\nforward_displacement_m={d[0]:.6f}\nlateral_displacement_m={d[1]:.6f}\nfinal_yaw_rad={yaw:.6f}\nfinal_base_height_m={s.d.qpos[2]:.6f}\nmin_base_height_m={minz:.6f}\nmean_abs_raw_torque_nm={x.mean():.6f}\nmax_abs_raw_torque_nm={x.max():.6f}\ntorque_over_13_ratio={(x>13).mean():.8f}\ntorque_over_15_ratio={(x>15).mean():.8f}\ntorque_over_limit_ratio={(x>s.limits).mean():.8f}")
if __name__=="__main__":main()
