"""Fresh free-joint IMU state after mj_step; legacy bridges remain reproducible.

mj_step integrates qpos/qvel but derived cvel still describes the pre-integration
state. Do not mix that angular velocity with post-integration joint q/dq in
leg odometry. Actor linear velocity remains sensor-only leg odometry.
"""
import mujoco
from sim2sim_v15 import V15Sim, get_parser, run
from sim2sim import quaternion_rotation_matrix


class SensorSyncSim(V15Sim):
    def base_velocity_world(self):
        joint = int(self.model.body_jntadr[self.trunk_body])
        if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected RS01 trunk free joint')
        dof = int(self.model.jnt_dofadr[joint])
        rotation = quaternion_rotation_matrix(self.data.qpos[3:7])
        # Free-joint translation is world-frame; angular velocity is body-frame.
        # Linear output is trunk origin velocity, not displaced COM velocity.
        velocity = self.data.qvel[dof:dof+6]
        return velocity[:3].copy(), rotation @ velocity[3:6]


if __name__ == '__main__':
    p=get_parser(); args=p.parse_args()
    if args.duration<=0 or args.settle_seconds<0 or not 0<=args.phase<1:
        p.error('Require duration>0, settle>=0, 0<=phase<1')
    run(args,sim_class=SensorSyncSim)
