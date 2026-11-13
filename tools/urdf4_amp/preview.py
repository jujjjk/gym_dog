"""Render measured motions; no policy inference or simulated controller is used here."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np,mujoco as mj
from PIL import Image,ImageDraw
from design import make_model,setup
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
manifest=json.loads((a.dataset/'manifest.json').read_text());clips=[x['clip'] for x in manifest['clips']];chosen=[n for n in ['forward','backward','left','right','turn_left','turn_right','forward_left','arc_left'] if n in clips];frames=[]
with tempfile.TemporaryDirectory() as temp:
 m=make_model(Path(temp));d=mj.MjData(m);qa,va,fb,_=setup(m);ground=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'ground');m.geom_group[ground]=2;m.geom_rgba[ground]=[.22,.25,.29,1];renderer=mj.Renderer(m,480,640);opt=mj.MjvOption();opt.geomgroup[0]=0
 for name in chosen:
  arr=np.load(a.dataset/(name+'.npz'));ref=next(x for x in manifest['clips'] if x['clip']==name)
  for i in range(0,150,5):
   d.qpos[:3]=arr['root_pos_world_m'][i];d.qpos[3:7]=arr['root_quat_wxyz'][i];d.qpos[qa]=arr['joint_pos_rad'][i];mj.mj_forward(m,d);cam=mj.MjvCamera();cam.lookat[:]=d.qpos[:3]+np.array([0,0,-.06]);cam.distance=.88;cam.azimuth=125;cam.elevation=-20;renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render()).resize((480,360));draw=ImageDraw.Draw(im);draw.rectangle((0,0,480,42),fill=(15,20,30));draw.text((10,7),'NEW URDF | '+name+' | physical recording',fill='white');draw.text((10,23),'Fresh reference + RS01 plant; not policy 5400',fill=(190,210,230));frames.append(im)
  frames[-15].save(a.output/(name+'.png'))
 renderer.close()
frames[0].save(a.output/'physical_motions.gif',save_all=True,append_images=frames[1:],duration=100,loop=0);print('Saved',a.output/'physical_motions.gif')
