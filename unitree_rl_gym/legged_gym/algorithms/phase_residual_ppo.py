"""Frozen locomotion backbone with a small phase-aware correction head."""

import torch
import torch.nn as nn

from rsl_rl.runners import OnPolicyRunner

from .conservative_ppo import ConservativePPO


class PhaseResidualActor(nn.Module):
    """Direct 12-action actor with a frozen base and bounded learned correction."""

    def __init__(
        self,
        base_actor,
        feature_indices,
        num_actions,
        hidden_dim=32,
        residual_scale=0.03,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.feature_indices = tuple(int(index) for index in feature_indices)
        self.residual_scale = float(residual_scale)
        for parameter in self.base_actor.parameters():
            parameter.requires_grad_(False)

        self.residual_actor = nn.Sequential(
            nn.Linear(len(self.feature_indices), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(num_actions)),
        )
        nn.init.orthogonal_(
            self.residual_actor[0].weight, gain=0.5
        )
        nn.init.zeros_(self.residual_actor[0].bias)
        # Exact policy parity at initialization.
        nn.init.zeros_(self.residual_actor[-1].weight)
        nn.init.zeros_(self.residual_actor[-1].bias)

    def forward(self, observations):
        base_actions = self.base_actor(observations)
        correction_features = observations[..., self.feature_indices]
        correction = self.residual_scale * torch.tanh(
            self.residual_actor(correction_features)
        )
        return base_actions + correction


class PhaseResidualOnPolicyRunner(OnPolicyRunner):
    """PPO runner that preserves a loaded gait and trains only its small head."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        actor_critic = self.alg.actor_critic
        actor_critic.actor = PhaseResidualActor(
            actor_critic.actor,
            feature_indices=self.cfg.get(
                "residual_feature_indices",
                [1, 5, 10, 11, 48, 49, 50, 51],
            ),
            num_actions=self.env.num_actions,
            hidden_dim=self.cfg.get("residual_hidden_dim", 32),
            residual_scale=self.cfg.get("residual_action_scale", 0.03),
        ).to(device)
        self.alg = ConservativePPO(
            actor_critic,
            device=device,
            reference_policy_coef=self.cfg.get(
                "reference_policy_coef", 0.5
            ),
            reference_action_deadband=self.cfg.get(
                "reference_action_deadband", 0.015
            ),
            reference_action_hinge_coef=self.cfg.get(
                "reference_action_hinge_coef", 8.0
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

    def load(self, path, load_optimizer=True):
        """Load either a standard base checkpoint or a residual checkpoint."""
        loaded = torch.load(path, map_location=self.device)
        loaded_state = loaded["model_state_dict"]
        if any(
            key.startswith("actor.base_actor.")
            for key in loaded_state
        ):
            self.alg.actor_critic.load_state_dict(loaded_state)
        else:
            current_state = self.alg.actor_critic.state_dict()
            mapped_state = {}
            for key, value in loaded_state.items():
                if key.startswith("actor."):
                    mapped_state[
                        "actor.base_actor." + key[len("actor."):]
                    ] = value
                else:
                    mapped_state[key] = value
            unexpected = sorted(set(mapped_state) - set(current_state))
            if unexpected:
                raise ValueError(
                    "Unexpected base checkpoint keys for residual actor: "
                    + ", ".join(unexpected)
                )
            current_state.update(mapped_state)
            self.alg.actor_critic.load_state_dict(current_state)
        if load_optimizer:
            if "optimizer_state_dict" not in loaded:
                raise ValueError("Checkpoint has no optimizer state")
            self.alg.optimizer.load_state_dict(
                loaded["optimizer_state_dict"]
            )
        self.current_learning_iteration = loaded.get("iter", 0)

    def set_reference_policy(self):
        self.alg.set_reference_policy()
