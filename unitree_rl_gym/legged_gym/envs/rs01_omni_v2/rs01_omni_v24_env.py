"""Replace pair-width cost with a contact/landing lateral corridor.

Truth is used ONLY for reward/diagnostics, never inserted into the 61D actor.
This is not a support-polygon/COM stability guarantee, or a hard joint mirror.
"""
import math
import torch
from isaacgym import gymapi, gymtorch
from .rs01_omni_v19_env import huber_positive
from .rs01_omni_v23_env import Rs01OmniV23Robot


def support_region_cost(signed_y, contact, touchdown, inner, outer, scale, touchdown_weight):
    per_foot = huber_positive((inner-signed_y)/scale) + huber_positive((signed_y-outer)/scale)
    weights = contact.float()*(1.+touchdown_weight*touchdown.float())
    return (per_foot*weights).sum(1)/contact.sum(1).clamp(min=1)


class Rs01OmniV24Robot(Rs01OmniV23Robot):
    def __init__(self, *args, **kwargs):
        self._v24_push_enabled = False
        super().__init__(*args, **kwargs)
        if self.cfg.domain_rand.push_robots:
            raise ValueError('V24 forbids legacy velocity-rewrite pushes; use queue_push')
        cfg=self.cfg.rewards
        if not 0 < cfg.support_inner_m < cfg.support_outer_m or cfg.support_region_scale_m <= 0:
            raise ValueError('Invalid support corridor')
        names=self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.v24_trunk_index=names.index('Trunk')
        if self.cfg.disturbance.audit_profiles and self.v24_trunk_index!=0:
            raise RuntimeError('Audit mass/COM expected Trunk as root body')
        self.v24_force_tensor=torch.zeros((self.num_envs,len(names),3),device=self.device)
        self.v24_push_vector=torch.zeros((self.num_envs,3),device=self.device)
        self.v24_push_ticks=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.v24_push_impulse=torch.zeros_like(self.v24_push_vector)
        # Verify that PhysX accepted requested creation-time properties.
        self.v24_verified_properties=[]
        for env_id,expected in getattr(self,'_v24_expected_properties',{}).items():
            props=self.gym.get_actor_rigid_body_properties(self.envs[env_id],self.actor_handles[env_id])
            p=props[self.v24_trunk_index]
            actual=(p.mass,p.com.x,p.com.y,p.com.z)
            if any(abs(a-b)>1e-5 for a,b in zip(actual,expected)):
                raise RuntimeError('PhysX did not retain mass/COM perturbation: %r != %r'%(actual,expected))
            self.v24_verified_properties.append(dict(env=env_id,mass_kg=p.mass,com_m=list(actual[1:])))

    def _process_rigid_body_props(self, props, env_id):
        props=super()._process_rigid_body_props(props,env_id)
        profiles=self.cfg.disturbance.audit_profiles
        if profiles:
            if len(profiles)!=self.num_envs:
                raise ValueError('Provide exactly one audit profile per environment')
            profile=profiles[env_id]
            # URDF root body is Trunk; verified against body names after creation.
            delta=float(profile.get('mass_delta_kg',0.))
            offset=profile.get('com_offset_m',(0.,0.,0.))
            if len(offset)!=3 or not all(math.isfinite(float(v)) for v in (delta,*offset)):
                raise ValueError('Invalid mass/COM profile')
            if props[0].mass+delta<=0:
                raise ValueError('Mass must remain positive')
            props[0].mass+=delta
            props[0].com.x+=float(offset[0]);props[0].com.y+=float(offset[1]);props[0].com.z+=float(offset[2])
            if not hasattr(self,'_v24_expected_properties'):self._v24_expected_properties={}
            p=props[0];self._v24_expected_properties[env_id]=(p.mass,p.com.x,p.com.y,p.com.z)
        return props

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        if hasattr(self,'v24_previous_contact'):
            self.v24_previous_contact[env_ids]=False
            self.v24_contact_valid[env_ids]=False
        if hasattr(self,'v24_push_ticks'):
            self.v24_push_ticks[env_ids]=0
            self.v24_push_vector[env_ids]=0
            # Do not erase delivered impulse telemetry when an episode fails.

    def _width_geometry_cost(self, widths, straight):
        ids=[self.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
        relative=self.feet_pos[:,ids]-self.root_states[:,None,:3]
        # Gravity-aligned yaw frame, so roll cannot shrink/expand the reference corridor.
        yaw=self.rpy[:,2]
        y=-torch.sin(yaw)[:,None]*relative[:,:,0]+torch.cos(yaw)[:,None]*relative[:,:,1]
        signed_y=y*y.new_tensor([1.,-1.,1.,-1.])
        contact=self.get_foot_contact_mask()[:,ids]
        if not hasattr(self,'v24_previous_contact'):
            self.v24_previous_contact=torch.zeros_like(contact)
            self.v24_contact_valid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        touchdown=contact & ~self.v24_previous_contact & self.v24_contact_valid[:,None]
        self.v24_previous_contact.copy_(contact);self.v24_contact_valid[:]=True
        cfg=self.cfg.rewards
        cost=support_region_cost(signed_y,contact,touchdown,cfg.support_inner_m,
                                cfg.support_outer_m,cfg.support_region_scale_m,cfg.touchdown_region_weight)
        self.v24_signed_y=signed_y.detach()
        self.v24_support_contact=contact.detach()
        self.v24_touchdown=touchdown.detach()
        self.v24_region_cost=cost.detach()
        self.v24_support_force=self.contact_forces[:,self.feet_indices[ids],2].clamp(min=0).detach()
        return cost  # Replaces old width cost; V21 coordination is added exactly once.

    def queue_push(self, env_ids, force_xyz_n, duration_s):
        if not math.isfinite(duration_s) or duration_s<=0:
            raise ValueError('Push duration must be positive and finite')
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long).reshape(-1)
        force=torch.as_tensor(force_xyz_n,device=self.device,dtype=torch.float).reshape(-1,3)
        if force.shape[0] not in (1,len(ids)) or not bool(torch.isfinite(force).all()):
            raise ValueError('Invalid force tensor')
        if bool((self.v24_push_ticks[ids]>0).any()):raise ValueError('Overlapping pulse')
        self.v24_push_vector[ids]=force
        self.v24_push_ticks[ids]=math.ceil(duration_s/self.cfg.sim.dt-1e-9)
        self._v24_push_enabled=True

    def _compute_torques(self, actions):
        result=super()._compute_torques(actions)
        if self._v24_push_enabled:
            self._apply_push_substep()
        return result

    def _apply_push_substep(self):
        applied=self.v24_push_vector*(self.v24_push_ticks>0)[:,None]
        self.v24_force_tensor.zero_()
        self.v24_force_tensor[:,self.v24_trunk_index]=applied
        ok=self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(self.v24_force_tensor),None,gymapi.ENV_SPACE)
        if not ok:raise RuntimeError('Failed to apply perturbation force')
        self.v24_push_impulse+=applied*self.cfg.sim.dt
        self.v24_push_ticks.sub_(1).clamp_(min=0)
