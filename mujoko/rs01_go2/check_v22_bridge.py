"""Same-state sensor effective-command / host-guard parity against training."""
from pathlib import Path
from types import SimpleNamespace
import sim2sim_v22 as b
import torch
import numpy as np
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_env import Rs01OmniV22Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_env import guarded_target

root=Path(__file__).resolve().parents[2]
sim=b.V22Sim(root/'artifacts/rs01_v20_sim2sim/scene.xml',root/'artifacts/rs01_v22_sim2sim/B23500.onnx',[0.,0.,0.])
fake=object.__new__(Rs01OmniV22Robot)
fake.sensor_ready=True;fake._direction_ready=True;fake.num_envs=1;fake.device='cpu';fake.cfg=Rs01OmniV22Cfg()
rng=np.random.default_rng(22)
maximum=0.
with torch.no_grad():
    for _ in range(1000):
        sim.command=rng.uniform([-.3,-.2,-.7],[.4,.2,.7]);sim.turning=bool(rng.integers(2))
        sim.heading_target=float(rng.uniform(-3,3));sim.sensor.yaw[:]=float(rng.uniform(-3,3))
        fake.sensor=sim.sensor;fake.commands=b.tensor(sim.command)
        fake.straight_heading_target_rad=torch.tensor([sim.heading_target]);fake.direction_turning=torch.tensor([sim.turning])
        expected=fake._direction_velocity_target()[0].numpy()
        maximum=max(maximum,float(abs(expected-sim.effective_command()).max()))
        np.testing.assert_allclose(expected,sim.effective_command(),atol=2e-6,rtol=0)
    # Exercise nontrivial sensor delay and motor feedback during actual movement.
    sim.set_command([.3,0.,0.])
    for _ in range(100):
        sim.update_policy()
        expected,_,_=guarded_target(b.tensor(sim.limited_target),sim.sensor.q,sim.sensor.dq,
            b.tensor(sim.kp),b.tensor(sim.kd),b.tensor(sim.guard_limit),b.tensor(sim.lower),b.tensor(sim.upper))
        # update_policy has advanced thermal limit; recompute with same current limit in both functions.
        actual,_,_=b.bridge.project_guard(sim.limited_target,sim.sensor.q[0].numpy(),sim.sensor.dq[0].numpy(),
            sim.kp,sim.kd,sim.guard_limit,sim.lower,sim.upper)
        np.testing.assert_allclose(expected[0].numpy(),actual,atol=3e-7,rtol=0)
        assert sim.observation().dtype==np.float32
        assert sim.sensor.imu_stamp[0]<=sim.feedback_tick and sim.sensor.motor_stamp[0]<=sim.feedback_tick
        for _ in range(sim.decimation):sim.capture();sim.physics_step()
        sim.capture();sim.sensor.read(sim.feedback_tick);sim.update_estimator()
print('PASS 1000 command cases, 100 moving guard cases; max_command_error=',maximum)
