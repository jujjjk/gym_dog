"""Initialize the existing61D->12D actor with a small physical-target phase prior.

No runtime CPG, qpos writes, teacher reward, old checkpoint, or actuator change.
The fitted network still sends targets through the original RS01 motor pipeline.
"""
import torch
from legged_gym.envs.rs01_go2_straight.rs01_odometry import Rs01TorchLegOdometry


def seed_targets(default,command,phase,period=.5,duty=.7,height=.025,mass_kg=None,kp=None):
    n=len(phase);device=phase.device
    fk=Rs01TorchLegOdometry(n,device)
    q=default.expand(n,-1).clone()
    feet,_=fk.foot_position_and_jacobian(q)
    local=(phase[:,None]+phase.new_tensor([0.,.5,.5,0.])).remainder(1.)
    swing=((local-duty)/(1.-duty)).clamp(0,1)
    smooth=swing.square()*swing*(10.-15.*swing+6.*swing.square())
    travel=torch.where(local<duty,.5-local/duty,-.5+smooth)
    velocity=command[:,:2,None].transpose(1,2).expand(-1,4,-1).clone()
    velocity[:,:,0]-=command[:,2,None]*feet[:,:,1]
    velocity[:,:,1]+=command[:,2,None]*feet[:,:,0]
    target=feet.clone();target[:,:,:2]+=travel[:,:,None]*period*duty*velocity
    target[:,:,2]+=height*torch.sin(torch.pi*swing).square()
    side=phase.new_tensor([1.,-1.,1.,-1.])
    target[:,:,1]=(target[:,:,1]*side).clamp(.135,.20)*side
    for _ in range(15):
        actual,jac=fk.foot_position_and_jacobian(q)
        step=torch.linalg.solve(jac,target.sub(actual).unsqueeze(-1)).squeeze(-1).clamp(-.12,.12)
        q+=step.reshape(n,12)
    actual,_=fk.foot_position_and_jacobian(q)
    if float((actual-target).abs().max())>1e-4:raise RuntimeError('Phase initializer IK failed')
    if mass_kg is not None:
        # Approximate load-induced PD deflection. This is a TARGET initialization,
        # not an extra force: all torque still goes through real motor limits.
        _,jac=fk.foot_position_and_jacobian(q)
        support=1.-torch.sin(torch.pi*swing).square()
        fz=mass_kg*9.81*support/support.sum(1,keepdim=True).clamp(min=1.)
        q-= (jac[:,:,2,:]*fz[:,:,None]).reshape(n,12)/kp
    return q


def initialize_actor(env,actor,steps=600):
    device=env.device;n=4096
    table=torch.tensor(env.cfg.stability.commands,device=device)
    command=table[torch.randint(len(table),(n,),device=device)]
    phase=torch.rand(n,device=device)
    # Compensate only the initialization prior's nominal target timing, not the plant.
    target=seed_targets(env.default_dof_pos,command,(phase+.06/env.cfg.rewards.gait_period_s).remainder(1.),
                        env.cfg.rewards.gait_period_s,env.cfg.rewards.gait_stance_ratio,
                        mass_kg=sum(p.mass for p in env.gym.get_actor_rigid_body_properties(env.envs[0],env.actor_handles[0])),
                        kp=env.p_gains)
    raw=(target-env.default_dof_pos)/env.rs01_action_scale_rad
    label=raw.clamp(-1,1)
    obs=env.get_observations()[torch.randint(env.num_envs,(n,),device=device)].clone()
    obs[:,9:12]=command*env.commands_scale;obs[:,57:60]=command*env.commands_scale
    obs[:,48]=torch.sin(2*torch.pi*phase);obs[:,49]=torch.cos(2*torch.pi*phase)
    obs[:,50]=0.;obs[:,51]=1.;obs[:,54]=1.
    obs[:,12:24]=torch.randn(n,12,device=device)*.06
    obs[:,24:36]=torch.randn(n,12,device=device)*.04
    obs[:,36:48]=torch.randn(n,12,device=device)*.25
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3)
    with torch.enable_grad():
        for _ in range(steps):
            ids=torch.randint(n,(512,),device=device)
            loss=(actor(obs[ids])-label[ids]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():mse=float((actor(obs)-label).square().mean())
    if not mse<.02:raise RuntimeError('Phase initializer failed to fit: '+str(mse))
    return dict(kind='actor_initialization_only_not_physical_validation',mse=mse,
                target_clipped_fraction=float((raw.abs()>1).float().mean()),
                clearance_m=.025,advance_s=.06,steps=steps,static_load_target_compensation=True)
