import sys
sys.path.insert(0,'/home/nszb/gym/unitree_rl_gym/legged_gym/scripts')
import play_urdf4_omni as play
from isaacgym import gymapi
import torch
import numpy as np
from pathlib import Path
from legged_gym.envs.urdf4.urdf4_omni_env import Urdf4OmniRobot
from legged_gym.utils import get_args
args=get_args();args.headless=True;args.task=play.TASK
original=Urdf4OmniRobot.create_sim
def create(self):
 self.graphics_device_id=self.sim_device_id
 return original(self)
Urdf4OmniRobot.create_sim=create
flags=type('Flags',(),{'disturbed':False})();env,cfg=play.make_env(args,flags,1);runner,path=play.load_policy(env,args,cfg);policy=runner.get_inference_policy(device=env.device)
cp=gymapi.CameraProperties();cp.width=640;cp.height=480;camera=env.gym.create_camera_sensor(env.envs[0],cp)
if camera<0:raise RuntimeError('camera creation failed')
env.set_command_override(torch.tensor([[args.vx,args.vy,args.wz]],device=env.device));obs,_=env.reset();frames=[]
with torch.no_grad():
 for k in range(round(args.duration_s/env.dt)):
  obs,*_=env.step(policy(obs))
  if k%5==0:
   env.gym.fetch_results(env.sim,True);b=env.root_states[0,:3].tolist();env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(b[0]+.95,b[1]-1.05,.65),gymapi.Vec3(b[0],b[1],.20));env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
   rgba=env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR).reshape(480,640,4);frames.append(rgba[:,:,:3].copy())
np.save('/home/nszb/gym/artifacts/urdf4_selection_20261113/preview_frames.npy',np.stack(frames));print('Saved preview frames',path,flush=True)
