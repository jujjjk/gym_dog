"""URDF4 AMP feature/contract adapter; no training task is registered here."""
import hashlib
import json
from pathlib import Path
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.algorithms.rs01_amp import StyleLearner

JOINT_ORDER = [leg + '_' + joint + '_joint'
               for leg in ('FL', 'FR', 'RL', 'RR')
               for joint in ('hip', 'thigh', 'calf')]
FOOT_ORDER = ['FL', 'FR', 'RL', 'RR']


def amp_features(env):
    """46D, unscaled simulator truth in canonical joint/foot order."""
    if list(env.dof_names) != JOINT_ORDER:
        raise ValueError('URDF4 AMP joint order mismatch')
    slots = [env.foot_legs.index(leg) for leg in FOOT_ORDER]
    q = env.root_states[:, 3:7]
    relative = env.feet_pos[:, slots] - env.root_states[:, None, :3]
    feet = quat_rotate_inverse(q[:, None].expand(-1, 4, -1).reshape(-1, 4),
                               relative.reshape(-1, 3)).reshape(env.num_envs, 12)
    down = q.new_tensor([0., 0., -1.]).repeat(env.num_envs, 1)
    return torch.cat((env.dof_pos, env.dof_vel, feet,
                      quat_rotate_inverse(q, env.root_states[:, 7:10]),
                      quat_rotate_inverse(q, env.root_states[:, 10:13]),
                      quat_rotate_inverse(q, down),
                      (env.root_states[:, 2] - env.env_origins[:, 2])[:, None]), 1)


def valid_transition(previous_reset, done, previous_command, next_command):
    return (~previous_reset.bool() & ~done.bool()
            & torch.isclose(previous_command, next_command, atol=1e-6, rtol=0).all(1))


class Urdf4AMPReference(StyleLearner):
    """Prevent same-named old-robot joints from bypassing URDF provenance checks."""
    def __init__(self, path, env, *, validation_only=False, settings=None, validation_receipt=None):
        manifest = json.loads((Path(path) / 'manifest.json').read_text())
        asset = Path(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        if hashlib.sha256(asset.read_bytes()).hexdigest() != manifest['urdf_sha256']:
            raise ValueError('AMP reference does not match the active URDF')
        if manifest['joint_order'] != JOINT_ORDER or list(env.dof_names) != JOINT_ORDER:
            raise ValueError('AMP joint order mismatch')
        if manifest['foot_order'] != FOOT_ORDER or manifest['quaternion_order'] != 'wxyz':
            raise ValueError('AMP foot/quaternion order mismatch')
        if manifest['feature_size'] != 46 or abs(env.dt - manifest['frame_dt_s']) > 1e-8:
            raise ValueError('AMP feature size or policy period mismatch')
        self.validation_only = bool(validation_only)
        receipt_verified = False
        if validation_receipt is not None:
            receipt = json.loads(Path(validation_receipt).read_text())
            required = ('nominal_isaac_reference_tracking_passed', 'isaac_fk_passed',
                        'AMP_discriminator_numerics_passed', 'one_AMP_PPO_integration_update_passed',
                        'dataset_file_hashes_match')
            if not all(receipt.get(k) is True for k in required):
                raise ValueError('Incomplete nominal Isaac AMP validation receipt')
            if receipt['urdf_sha256'] != manifest['urdf_sha256'] or receipt['dataset_manifest_sha256'] != hashlib.sha256((Path(path)/'manifest.json').read_bytes()).hexdigest():
                raise ValueError('Validation receipt belongs to another URDF/dataset')
            for clip in manifest['clips']:
                file = Path(path)/(clip['clip']+'.npz')
                if hashlib.sha256(file.read_bytes()).hexdigest() != clip['sha256']:
                    raise ValueError('Validated AMP clip has changed: '+clip['clip'])
            receipt_verified = True
        super().__init__(str(path), device=env.device,
                         allow_kinematic=validation_only or receipt_verified, settings=settings)
