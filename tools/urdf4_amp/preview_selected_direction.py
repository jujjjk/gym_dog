"""Open the original Gym viewer and capture a short walking preview, read only."""
import argparse,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
from legged_gym.envs.urdf4.amp_direction_env import Urdf4AMPDirectionRobot

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--selection',type=Path,default=ROOT/'artifacts/urdf4_direction_selection_20261116/selected.json')
p.add_argument('--frames',type=Path,default=ROOT/'artifacts/urdf4_direction_selection_20261116/preview_frames')
own,rest=p.parse_known_args()
selection=json.loads(own.selection.read_text());checkpoint=Path(selection['checkpoint'])
own.frames.mkdir(parents=True,exist_ok=True)
original=Urdf4AMPDirectionRobot.step
counter=[0]
def capture(env,actions):
    result=original(env,actions)
    k=counter[0];counter[0]+=1
    if env.viewer is not None and 150<=k<350 and (k-150)%5==0:
        path=own.frames/('frame_%03d.png'%((k-150)//5))
        env.gym.write_viewer_image_to_file(env.viewer,str(path))
    return result
Urdf4AMPDirectionRobot.step=capture
sys.argv=['play.py','--task','urdf4_amp_cts_direction_v1','--load_run',str(checkpoint.parent),
    '--checkpoint',str(selection['selected']['iteration']),'--num_envs','1','--vx','.25','--vy','0','--wz','0',
    '--duration_s','120','--sim_device','cuda:1','--rl_device','cuda:1']+rest
runpy.run_path(str(ROOT/'unitree_rl_gym/legged_gym/scripts/play.py'),run_name='__main__')
