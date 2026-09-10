"""Bridge velocities must correspond to current qpos/qvel, not cached cvel."""
import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mujoko/rs01_go2'))
from sim2sim_sensor_sync import SensorSyncSim
from sim2sim import quaternion_rotation_matrix


def test_free_joint_sensor_state_matches_fresh_jacobian_not_cached_cvel():
    sim=object.__new__(SensorSyncSim)
    sim.model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.0025"/>
    <worldbody><body name="Trunk" pos="0 0 1"><freejoint/>
    <inertial pos="0 .01 -.0074" quat=".5 .5 -.5 .5" mass="12" diaginertia=".1 .2 .3"/>
    <geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    sim.data=mujoco.MjData(sim.model);sim.trunk_body=1
    rng=np.random.default_rng(16)
    for _ in range(30):
        q=rng.normal(size=4);q/=np.linalg.norm(q)
        sim.data.qpos[3:7]=q;sim.data.qvel[:]=rng.normal(size=6)
        mujoco.mj_forward(sim.model,sim.data)
        # Deliberately change velocity after the cached velocity was computed.
        sim.data.qvel[:]+=rng.normal(size=6)
        before=sim.data.qvel.copy()
        linear,angular=sim.base_velocity_world()
        fresh=mujoco.MjData(sim.model)
        fresh.qpos[:]=sim.data.qpos;fresh.qvel[:]=sim.data.qvel
        mujoco.mj_forward(sim.model,fresh)
        jp=np.zeros((3,6));jr=np.zeros((3,6))
        mujoco.mj_jac(sim.model,fresh,jp,jr,fresh.xpos[1],1)
        np.testing.assert_allclose(linear,jp@fresh.qvel,atol=1e-12)
        np.testing.assert_allclose(angular,jr@fresh.qvel,atol=1e-12)
        np.testing.assert_array_equal(sim.data.qvel,before)
        np.testing.assert_allclose(sim.base_velocity_body()[1],before[3:6],atol=1e-12)
