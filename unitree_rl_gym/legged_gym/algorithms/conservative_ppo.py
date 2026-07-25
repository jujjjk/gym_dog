"""PPO continuation constrained to stay near a selected reference policy."""

import copy

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner


def executed_action_delta(policy_mean, reference_mean):
    """Difference in the normalized action space executed by the environment."""
    return torch.tanh(policy_mean) - torch.tanh(reference_mean)


class ConservativePPO(PPO):
    """Standard PPO with a fixed-reference deterministic-action trust region."""

    def __init__(
        self,
        *args,
        reference_policy_coef=0.25,
        reference_action_deadband=0.10,
        reference_action_hinge_coef=2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_policy_coef = float(reference_policy_coef)
        self.reference_action_deadband = float(reference_action_deadband)
        self.reference_action_hinge_coef = float(reference_action_hinge_coef)
        self.reference_actor = None

    def set_reference_policy(self):
        """Freeze the currently loaded actor as the continuation anchor."""
        self.reference_actor = copy.deepcopy(self.actor_critic.actor)
        self.reference_actor.to(self.device)
        self.reference_actor.eval()
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)

    def update(self):
        if self.reference_actor is None:
            raise RuntimeError(
                "ConservativePPO reference policy was not initialized"
            )
        if self.actor_critic.is_recurrent:
            raise NotImplementedError(
                "ConservativePPO currently supports MLP policies only"
            )

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            entropy_batch = self.actor_critic.entropy

            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = (
                -torch.squeeze(advantages_batch)
                * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
            )
            surrogate_loss = torch.max(
                surrogate, surrogate_clipped
            ).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = torch.square(value_batch - returns_batch)
                value_losses_clipped = torch.square(
                    value_clipped - returns_batch
                )
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = torch.square(
                    returns_batch - value_batch
                ).mean()

            with torch.no_grad():
                reference_mu = self.reference_actor(obs_batch)
            reference_delta = executed_action_delta(
                mu_batch, reference_mu
            )
            reference_loss = torch.mean(torch.square(reference_delta))
            reference_hinge_loss = torch.mean(torch.square(
                (
                    torch.abs(reference_delta)
                    - self.reference_action_deadband
                ).clip(min=0.0)
            ))
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.reference_policy_coef * reference_loss
                + self.reference_action_hinge_coef * reference_hinge_loss
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


class ConservativeOnPolicyRunner(OnPolicyRunner):
    """Run ConservativePPO while retaining standard checkpoint compatibility."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        actor_critic = self.alg.actor_critic
        self.alg = ConservativePPO(
            actor_critic,
            device=device,
            reference_policy_coef=self.cfg.get(
                "reference_policy_coef", 0.25
            ),
            reference_action_deadband=self.cfg.get(
                "reference_action_deadband", 0.10
            ),
            reference_action_hinge_coef=self.cfg.get(
                "reference_action_hinge_coef", 2.0
            ),
            **self.alg_cfg,
        )
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

    def set_reference_policy(self):
        self.alg.set_reference_policy()
