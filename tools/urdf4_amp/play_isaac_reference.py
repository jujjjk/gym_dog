"""Native Isaac Gym physics viewer for validated V2 teacher commands.

Not a learned AMP policy and not pose playback. States are initialized only at
motion boundaries; every displayed motion then runs through the RS01 plant.
"""
import argparse,json,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import play_urdf4_omni as play
from isaacgym import gymapi,gymtorch
import numpy as np
import torch
from legged_gym.utils import get_args,task_registry

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2')
p.add_argument('--commands-dir',type=Path,default=ROOT/'artifacts/urdf4_amp_isaac_validation_20261113/commands60')
p.add_argument('--clip',default='forward');p.add_argument('--sweep',action='store_true')
p.add_argument('--seconds',type=float,default=8);p.add_argument('--follow',action='store_true')
p.add_argument('--cycles',type=int,default=0,help='0 repeats until the window is closed')
own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;args=get_args();args.task=play.TASK;args.num_envs=1
if own.seconds<=0:p.error('--seconds must be positive')
m=json.loads((own.dataset/'manifest.json').read_text());clips={c['clip']:c for c in m['clips']}
if own.clip not in clips:p.error('Unknown motion: '+own.clip)
names=list(clips) if own.sweep else [own.clip];pivot=names.index(own.clip);names=names[pivot:]+names[:pivot]
data={name:dict(np.load(own.commands_dir/name/'physical.npz',allow_pickle=False)) for name in names}
cfg,_=task_registry.get_cfgs(name=play.TASK);cfg.env.num_envs=1;cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10;cfg.noise.add_noise=False;cfg.push.enabled=False;cfg.domain_rand.push_robots=False;cfg.commands.fixed_command=[0,0,0]
for flag in ['randomize_friction','randomize_restitution','randomize_base_mass','randomize_base_com','randomize_link_mass','randomize_armature']:setattr(cfg.domain_rand,flag,False)
env,_=task_registry.make_env(name=play.TASK,args=args,env_cfg=cfg);assert list(env.dof_names)==m['joint_order'];device=env.device;original_reset=env.reset_idx
state={'quit':False,'paused':False,'delta':0,'restart':False,'follow':own.follow,'failed':False};window_id=None
if env.viewer:
 for key,event in [(gymapi.KEY_N,'next'),(gymapi.KEY_B,'previous'),(gymapi.KEY_R,'restart'),(gymapi.KEY_SPACE,'pause'),(gymapi.KEY_F,'follow')]:env.gym.subscribe_viewer_keyboard_event(env.viewer,key,event)
 # Own the native render event loop so key events aren't consumed by BaseTask.
 env.render=lambda *a,**kw:None
 if shutil.which('xwininfo') and shutil.which('xprop'):
  tree=subprocess.run(['xwininfo','-root','-tree'],capture_output=True,text=True,encoding='utf-8',errors='replace').stdout
  found=re.findall(r'(0x[0-9a-fA-F]+) "Isaac Gym"',tree)
  if found:window_id=found[-1]
def title(name,k=0):
 if not window_id:return
 text='Isaac Gym | URDF4 AMP V2 | '+name+' | N/B next/prev, Space pause, F follow'
 for prop,fmt in [('_NET_WM_NAME','8u'),('WM_NAME','8s')]:subprocess.run(['xprop','-id',window_id,'-f',prop,fmt,'-set',prop,text],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def fail(ids):
 if len(ids) and not state['failed']:
  state['failed']=True;state['paused']=True;print('[failure] Termination detected; paused without reset. N/B switches, R restarts.',flush=True)
def initialize(name):
 z=data[name];q0=torch.tensor(z['actuator_default'],device=device,dtype=torch.float)[None];root=torch.tensor(z['initial_root'],device=device,dtype=torch.float)[None];command=torch.tensor(clips[name]['command'],device=device,dtype=torch.float)[None];env.reset_idx=original_reset;env.set_command_override(command);env.reset();env.reset_idx=fail
 env.dof_pos[:]=q0;env.dof_vel[:]=0;env.root_states[:,:3]=env.env_origins+root[:,:3];env.root_states[:,3:7]=root[:,[4,5,6,3]];env.root_states[:,7:]=0;correction=(q0-env.default_dof_pos)*(1/.928-1);target=q0+correction;env.position_target[:]=target;env.response_target[:]=q0;env.target_rate[:]=0;env.delay_buffer[:]=target;env.response_tick=0;env.raw_pd_torques[:]=0;env.motor_torques[:]=0;env.episode_length_buf[:]=0;env.common_step_counter=0;env.episode_steps[:]=0;env.actions[:]=0;env.last_actions[:]=0;env.foot_contact_mask[:]=False
 env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_states));env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state));env.gym.refresh_actor_root_state_tensor(env.sim);env.gym.refresh_dof_state_tensor(env.sim);env.gym.refresh_rigid_body_state_tensor(env.sim);env._refresh_feet();env._reset_sensors(torch.arange(1,device=device));env.compute_observations();state['failed']=False;title(name)
 print('[physics motion]',name,'command',clips[name]['command'],'| 2s settle, then',own.seconds,'s | no pose writes during motion',flush=True)
 return torch.tensor(z['command_targets'],device=device,dtype=torch.float),correction

def events():
 if not env.viewer:return
 if env.gym.query_viewer_has_closed(env.viewer):state['quit']=True;return
 for event in env.gym.query_viewer_action_events(env.viewer):
  if event.value<=0:continue
  if event.action=='QUIT':state['quit']=True
  elif event.action=='next':state['delta']+=1
  elif event.action=='previous':state['delta']-=1
  elif event.action=='restart':state['restart']=True
  elif event.action=='pause':state['paused']=not state['paused']
  elif event.action=='follow':state['follow']=not state['follow'];print('[camera] follow',state['follow'],flush=True)
  elif event.action=='toggle_viewer_sync':env.enable_viewer_sync=not env.enable_viewer_sync

def draw(delta):
 if not env.viewer:return
 env.gym.fetch_results(env.sim,True)
 if state['follow'] and np.linalg.norm(delta)>1e-9:
  camera=env.gym.get_viewer_camera_transform(env.viewer,env.envs[0]);direction=camera.r.rotate(gymapi.Vec3(0,0,1));eye=gymapi.Vec3(camera.p.x+float(delta[0]),camera.p.y+float(delta[1]),camera.p.z);target=gymapi.Vec3(eye.x+direction.x,eye.y+direction.y,eye.z+direction.z);env.gym.viewer_camera_look_at(env.viewer,env.envs[0],eye,target)
 env.gym.step_graphics(env.sim);env.gym.draw_viewer(env.viewer,env.sim,True)
 if env.enable_viewer_sync and not state['paused']:env.gym.sync_frame_time(env.sim)
 if state['paused']:time.sleep(.02)

index,k,completed=0,0,0;targets,correction=initialize(names[index]);previous_xy=env.root_states[0,:2].cpu().numpy().copy()
if env.viewer:env.set_camera([1.,-1.35,.75],[0,0,.20])
with torch.no_grad():
 while not state['quit']:
  events()
  if state['quit']:break
  boundary=k>=min(len(targets),round((own.seconds+2)/env.dt))
  if state['delta'] or state['restart'] or boundary:
   if boundary:
    completed+=1
    if own.cycles and completed>=own.cycles*len(names):break
   new_index=(index+(state['delta'] if state['delta'] else (0 if state['restart'] else 1)))%len(names);old_xy=env.root_states[0,:2].cpu().numpy().copy();index=new_index;targets,correction=initialize(names[index]);k=0;state['delta']=0;state['restart']=False;state['paused']=False;new_xy=env.root_states[0,:2].cpu().numpy().copy();draw(new_xy-old_xy);previous_xy=new_xy
  if not state['paused']:
   desired=targets[k][None]+correction;actions=(desired-env.default_dof_pos)/env.action_scale;env.step(actions);k+=1
  new_xy=env.root_states[0,:2].cpu().numpy().copy();draw(new_xy-previous_xy);previous_xy=new_xy
print('[done]',completed,'motions',flush=True)
if env.viewer:env.gym.destroy_viewer(env.viewer)
env.gym.destroy_sim(env.sim)
