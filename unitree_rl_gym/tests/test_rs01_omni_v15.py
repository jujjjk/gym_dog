import isaacgym
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_config import Rs01OmniV15SupportCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v14_config import Rs01OmniV14ActuatorCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_env import Rs01OmniV15Robot
from legged_gym.envs.rs01_go2_straight.rs01_odometry import Rs01TorchLegOdometry
from legged_gym.envs.rs01_go2_straight.rs01_support_odometry import GravitySupportOdometry
from legged_gym.utils.helpers import class_to_dict


def test_contract_unchanged_except_phase_and_estimator():
    old, new = Rs01OmniV14ActuatorCfg(), Rs01OmniV15SupportCfg()
    for key in ('rewards', 'commands', 'control', 'sim', 'rs01_actuator', 'rs01_odometry', 'domain_rand'):
        assert class_to_dict(getattr(old, key)) == class_to_dict(getattr(new, key)), key


def test_stand_freezes_march_resumes_without_phase_reset():
    r = object.__new__(Rs01OmniV15Robot)
    r.dt = .02
    r.wide_gait_phase = torch.tensor([.25, .75])
    r.gait_enable = torch.tensor([0., 1.])
    r._command_gait_frequency = lambda: torch.tensor([2., 2.])
    r._advance_wide_gait_phase()
    assert torch.allclose(r.wide_gait_phase, torch.tensor([.25, .79]))
    r.gait_enable[:] = torch.tensor([1., 0.])
    r._advance_wide_gait_phase()
    assert torch.allclose(r.wide_gait_phase, torch.tensor([.29, .79]))


def test_sensor_estimator_support_and_reset():
    core = Rs01TorchLegOdometry(2, 'cpu')
    estimator = GravitySupportOdometry(core)
    estimator.gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    # Synthetic FK isolates estimator math, including lateral support velocity.
    p = torch.tensor([[[.2,.1,-.291],[.2,-.1,-.291],[-.2,.1,-.291],[-.2,-.1,-.291]]]).repeat(2,1,1)
    core.foot_position_and_jacobian = lambda q: (p, torch.eye(3).repeat(2,4,1,1))
    dq = torch.tensor([0., -.1, 0.]).repeat(2,4,1)
    for _ in range(30): result = estimator.estimate(torch.zeros(2,12), dq, torch.zeros(2,3))
    assert torch.allclose(result['base_linear_velocity'][:,1], torch.full((2,),.1), atol=1e-5)
    assert result['stance_mask'].all()
    # All feet too high: no fictitious high-confidence support.
    p[:,:,2] = -.1
    result = estimator.estimate(torch.zeros(2,12), dq, torch.zeros(2,3))
    assert not result['confidence'].any()
    estimator.reset(torch.tensor([0]))
    assert not core.filtered[0].any() and core.filtered[1,1] > 0


def test_cpu_gpu_estimator_parity():
    torch.manual_seed(15)
    q = torch.tensor([0.,-.328,1.31]).repeat(16,4) + .04*torch.randn(16,12)
    dq = .1*torch.randn(16,12); omega = .05*torch.randn(16,3)
    results=[]
    for device in ('cpu','cuda:0'):
        estimator=GravitySupportOdometry(Rs01TorchLegOdometry(16,device))
        estimator.gravity=torch.tensor([[.03,.02,-1.]]).repeat(16,1)
        for _ in range(10): out=estimator.estimate(q,dq,omega)
        results.append(out)
    for key in ('base_linear_velocity','confidence','stance_mask'):
        assert torch.allclose(results[0][key].float(),results[1][key].cpu().float(),atol=1e-6),key
