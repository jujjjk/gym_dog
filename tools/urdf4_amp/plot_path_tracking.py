"""Standalone scientific plot; run in a plotting venv, not the Gym NumPy env."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();data=np.load(str(a.input));time=data['time'];end=int(np.searchsorted(time,60.))
# soft_nominal60 uses speed-major [off,heading,path], four envs per case.
ids=[12,16,20];labels=['Original frozen policy','Heading feedback','Heading + cross-track'];colors=['#c44e52','#dd8b27','#228b62']
fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
for idx,label,color in zip(ids,labels,colors):
    xy=data['xy'][:end,idx]
    axes[0].plot(xy[:,0],xy[:,1],label=label,color=color,lw=1.5)
    axes[2].plot(time[:end],abs(xy[:,1])*100,color=color,label=label,lw=1.3)
xy=data['xy'][:end,20];axes[1].axhspan(-5,5,color='#d7eedc',alpha=.7,label='5 cm corridor')
axes[1].plot(xy[:,0],xy[:,1]*100,color=colors[2],lw=1.5);axes[1].axhline(0,color='black',ls='--',lw=.7)
axes[1].set_ylim(-6,6)
axes[2].set_yscale('symlog',linthresh=5);axes[2].axhline(5,color='black',ls='--',lw=.8)
axes[0].set(title='Ground-truth XY paths',xlabel='X (m)',ylabel='Y (m)')
axes[1].set(title='Closed-loop path: lateral zoom',xlabel='X (m)',ylabel='Y (cm)')
axes[2].set(title='Absolute lateral deviation',xlabel='Time (s)',ylabel='Deviation (cm; linear below 5)')
for ax in axes:ax.grid(alpha=.2)
axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
fig.suptitle('Frozen AMP+CTS | command vx=0.25 m/s | seed 1 | ideal external pose (diagnostic)',fontsize=12)
a.output.parent.mkdir(exist_ok=True,parents=True)
fig.savefig(str(a.output.with_suffix('.png')),dpi=180);fig.savefig(str(a.output.with_suffix('.pdf')))
print(a.output.with_suffix('.png'))
