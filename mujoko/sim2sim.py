from pathlib import Path
import argparse,time,mujoco,numpy as np,onnxruntime as ort

J=[f"{l}_{j}_joint" for l in ("FL","FR","RL","RR") for j in ("hip","thigh","calf")]
Q=np.array([0,.563,-.95]*4,np.float32);KP=np.array([60,70,70]*4,np.float32);KD=np.array([.6,.8,.8]*4,np.float32)
S=np.array([.08,.18,.18,.08,.18,.18,.08,.20,.20,.08,.20,.20],np.float32);PO=np.array([0,.5,.5,0],np.float32)
def gravity(q):
    r=np.empty(9);mujoco.mju_quat2Mat(r,q);return r.reshape(3,3).T@np.array([0,0,-1.])
def gait(ph):
    p=(ph+PO)%1;s=np.clip((p-.62)/.38,0,1);s=s*s*(3-2*s);o=np.zeros(12,np.float32);o[2::3]=-.3*np.sin(np.pi*s)*(p>=.62);return o
class Sim:
    def __init__(self,model,policy,cmd,heading_hold=False):
        self.m=mujoco.MjModel.from_xml_path(str(model));self.d=mujoco.MjData(self.m);self.net=ort.InferenceSession(str(policy));self.cmd=np.array(cmd,np.float32);self.heading_hold=heading_hold
        self.q=np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in J]);self.v=np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,x)] for x in J]);self.aid=np.array([mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,x+"_motor") for x in J]);self.bid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,"Trunk");self.reset()
    def reset(self):
        mujoco.mj_resetData(self.m,self.d);self.d.qpos[:7]=[0,0,.295,1,0,0,0];self.d.qpos[self.q]=Q;self.action=np.zeros(12,np.float32);self.target=Q.copy();self.n=0;mujoco.mj_forward(self.m,self.d)
    def policy(self):
        vel=np.empty(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.bid,vel,1);ph=(self.n*.02%.54)/.54
        quat=self.d.qpos[3:7];yaw=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
        command=self.cmd.copy()
        if self.heading_hold:
            heading_error=np.arctan2(np.sin(-yaw),np.cos(-yaw));command[2]=np.clip(.5*heading_error,-1,1)
        obs=np.concatenate((vel[3:]*2,vel[:3]*.25,gravity(quat),command*np.array([2,2,.25]),self.d.qpos[self.q]-Q,self.d.qvel[self.v]*.05,self.action,[np.sin(2*np.pi*ph),np.cos(2*np.pi*ph)])).astype(np.float32)
        self.action=np.tanh(self.net.run(["raw_actions"],{"observations":np.clip(obs,-100,100)[None]})[0][0]);self.target=Q+S*self.action+gait(ph);self.n+=1
    def step(self):
        raw=KP*(self.target-self.d.qpos[self.q])-KD*self.d.qvel[self.v]
        self.d.ctrl[self.aid]=np.clip(raw,-17,17);mujoco.mj_step(self.m,self.d);return raw
def main():
    root=Path(__file__).parent;p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=root/"models/fanfan_scene.xml");p.add_argument("--policy",type=Path,default=root/"models/fanfan_best.onnx");p.add_argument("--duration",type=float,default=20);p.add_argument("--command",nargs=3,type=float,default=[.225,0,0]);p.add_argument("--heading-hold",action="store_true");p.add_argument("--viewer",action="store_true");a=p.parse_args();s=Sim(a.model,a.policy,a.command,a.heading_hold);dec=round(.02/s.m.opt.timestep)
    if a.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(s.m,s.d) as v:
            v.cam.distance=1.4;v.cam.azimuth=135;v.cam.elevation=-18;t=time.time();episode_start=t;i=0
            while v.is_running() and time.time()-t<a.duration:
                st=time.time()
                if st-episode_start>=20.0:
                    s.reset();episode_start=st;i=0
                if i%dec==0:s.policy()
                s.step();v.cam.lookat[:]=s.d.qpos[:3];v.sync();i+=1;time.sleep(max(0,s.m.opt.timestep-(time.time()-st)))
    else:
        raw=[];start=s.d.qpos[:2].copy();minz=99
        for i in range(int(a.duration/s.m.opt.timestep)):
            if i%dec==0:s.policy()
            raw.append(s.step());minz=min(minz,s.d.qpos[2])
        x=np.abs(raw);q=s.d.qpos[3:7];yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));d=s.d.qpos[:2]-start
        print(f"duration_s={a.duration:.3f}\nforward_displacement_m={d[0]:.6f}\nlateral_displacement_m={d[1]:.6f}\nfinal_yaw_rad={yaw:.6f}\nfinal_base_height_m={s.d.qpos[2]:.6f}\nmin_base_height_m={minz:.6f}\nmean_abs_raw_torque_nm={x.mean():.6f}\nmax_abs_raw_torque_nm={x.max():.6f}\ntorque_over_13_ratio={(x>13).mean():.8f}\ntorque_over_15_ratio={(x>15).mean():.8f}\ntorque_over_17_ratio={(x>17).mean():.8f}")
if __name__=="__main__":main()
