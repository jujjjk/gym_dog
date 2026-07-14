"""PPO extension that makes the locomotion actor left/right equivariant."""

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner


def mirror_fanfan_observations(observations):
    """Reflect Fanfan's 52-element observation across its sagittal plane."""
    if observations.shape[-1] != 52:
        raise ValueError(
            f"Fanfan symmetry expects 52 observations, got {observations.shape[-1]}"
        )
    mirrored = observations.clone()

    # Polar vectors: body linear velocity, gravity, and velocity command.
    mirrored[..., 1] *= -1.0
    mirrored[..., 7] *= -1.0
    mirrored[..., 10] *= -1.0
    # Axial body angular velocity and yaw-rate command.
    mirrored[..., 3] *= -1.0
    mirrored[..., 5] *= -1.0
    mirrored[..., 11] *= -1.0

    # Joint positions, velocities, and previous actions use policy joint order:
    # FL, FR, RL, RR, with hip/thigh/calf for each leg.
    leg_mirror = (1, 0, 3, 2)
    joint_sign = observations.new_tensor((-1.0, 1.0, 1.0))
    for start in (12, 24, 36):
        block = observations[..., start:start + 12].reshape(
            *observations.shape[:-1], 4, 3
        )
        block = block[..., leg_mirror, :] * joint_sign
        mirrored[..., start:start + 12] = block.reshape(
            *observations.shape[:-1], 12
        )

    # A left/right reflection swaps the two trot diagonals (half-cycle shift).
    mirrored[..., 48:50] *= -1.0
    # Heading error is represented as sin(error), cos(error).
    mirrored[..., 50] *= -1.0
    return mirrored


def mirror_fanfan_actions(actions):
    """Reflect policy-order joint actions across the sagittal plane."""
    if actions.shape[-1] != 12:
        raise ValueError(
            f"Fanfan symmetry expects 12 actions, got {actions.shape[-1]}"
        )
    leg_mirror = (1, 0, 3, 2)
    joint_sign = actions.new_tensor((-1.0, 1.0, 1.0))
    block = actions.reshape(*actions.shape[:-1], 4, 3)
    return (block[..., leg_mirror, :] * joint_sign).reshape(
        *actions.shape[:-1], 12
    )


class SymmetryPPO(PPO):
    """Standard PPO plus a deterministic mirrored-policy consistency loss."""

    def __init__(self, *args, symmetry_coef=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetry_coef = symmetry_coef

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        if self.actor_critic.is_recurrent:
            raise NotImplementedError("SymmetryPPO currently supports MLP policies only")
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, hid_states_batch,
             masks_batch) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        ) / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            mirrored_obs = mirror_fanfan_observations(obs_batch)
            mirrored_mu = self.actor_critic.actor(mirrored_obs)
            symmetry_loss = torch.mean(
                torch.square(mirrored_mu - mirror_fanfan_actions(mu_batch))
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.symmetry_coef * symmetry_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss


class SymmetryOnPolicyRunner(OnPolicyRunner):
    """Use SymmetryPPO while retaining the stock runner and checkpoint format."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        actor_critic = self.alg.actor_critic
        self.alg = SymmetryPPO(
            actor_critic,
            device=device,
            symmetry_coef=self.cfg.get("symmetry_coef", 0.5),
            **self.alg_cfg,
        )
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )
