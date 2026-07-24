import os, re
from datetime import datetime
from typing import Tuple
import torch, numpy as np, sys
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from legged_gym.algorithms import (
    ConservativeOnPolicyRunner,
    PhaseResidualOnPolicyRunner,
    SymmetryOnPolicyRunner,
)
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    get_load_path,
    set_seed,
    parse_sim_params,
)
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TaskRegistry:

    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[(LeggedRobotCfg, LeggedRobotCfgPPO)]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        env_cfg.seed = train_cfg.seed
        return (env_cfg, train_cfg)

    def make_env(
        self, name, args=None, env_cfg=None
    ) -> Tuple[(VecEnv, LeggedRobotCfg)]:
        """Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name'

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        if args is None:
            args = get_args()
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            (env_cfg, _) = self.get_cfgs(name)
        (env_cfg, _) = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        sim_params = {"sim": (class_to_dict(env_cfg.sim))}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=(args.physics_engine),
            sim_device=(args.sim_device),
            headless=(args.headless),
        )
        return (env, env_cfg)

    def make_alg_runner(
        self, env, name=None, args=None, train_cfg=None, log_root="default"
    ) -> Tuple[(OnPolicyRunner, LeggedRobotCfgPPO)]:
        """Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            (_, train_cfg) = self.get_cfgs(name)
        elif name is not None:
            print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        (_, train_cfg) = update_cfg_from_args(None, train_cfg, args)
        if log_root == "default":
            log_root = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
            )
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name,
            )
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name,
            )
        train_cfg_dict = class_to_dict(train_cfg)
        if train_cfg_dict["runner"].get("phase_residual_policy", False):
            runner_class = PhaseResidualOnPolicyRunner
        elif train_cfg_dict["runner"].get("reference_policy_coef", 0.0) > 0.0:
            runner_class = ConservativeOnPolicyRunner
        elif train_cfg_dict["runner"].get("symmetry_coef", 0.0) > 0.0:
            runner_class = SymmetryOnPolicyRunner
        else:
            runner_class = OnPolicyRunner
        runner = runner_class(env, train_cfg_dict, log_dir, device=(args.rl_device))
        actor_output_init_scale = train_cfg_dict["runner"].get(
            "actor_output_init_scale", None
        )
        if actor_output_init_scale is not None:
            output_layer = runner.alg.actor_critic.actor[-1]
            if not isinstance(output_layer, torch.nn.Linear):
                raise TypeError(
                    "actor_output_init_scale requires a linear actor output layer"
                )
            scale = float(actor_output_init_scale)
            torch.nn.init.uniform_(output_layer.weight, -scale, scale)
            torch.nn.init.zeros_(output_layer.bias)
        resume = train_cfg.runner.resume
        if resume:
            resume_path = get_load_path(
                log_root,
                load_run=(train_cfg.runner.load_run),
                checkpoint=(train_cfg.runner.checkpoint),
            )
            print(f"Loading model from: {resume_path}")
            load_optimizer = getattr(train_cfg.runner, "load_optimizer", True)
            if getattr(train_cfg.runner, "adapt_observation_input", False):
                loaded = torch.load(resume_path, map_location=(args.rl_device))
                loaded_state = loaded["model_state_dict"]
                current_state = runner.alg.actor_critic.state_dict()
                adapted = []
                for key in ("actor.0.weight", "critic.0.weight"):
                    old_weight = loaded_state[key]
                    new_weight = current_state[key]
                    if old_weight.shape == new_weight.shape:
                        pass
                    else:
                        if (
                            old_weight.ndim != 2
                            or new_weight.ndim != 2
                            or new_weight.ndim != 2
                            or old_weight.shape[1] > new_weight.shape[1]
                        ):
                            raise ValueError(
                                f"Observation adapter only supports widening the first input layer, got {key}: {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}"
                            )
                        widened = torch.zeros_like(new_weight)
                        widened[:, : old_weight.shape[1]] = old_weight
                        loaded_state[key] = widened
                        adapted.append(
                            f"{key} {old_weight.shape[1]}->{new_weight.shape[1]}"
                        )
                    runner.alg.actor_critic.load_state_dict(loaded_state)

                if load_optimizer:
                    raise ValueError(
                        "Optimizer state cannot be loaded after widening the observation input"
                    )
                runner.current_learning_iteration = loaded.get("iter", 0)
                print(
                    "Adapted checkpoint observation input: "
                    + (", ".join(adapted) if adapted else "not needed")
                )
            else:
                runner.load(resume_path, load_optimizer=load_optimizer)
            checkpoint_match = re.fullmatch(
                "model_(\\d+)\\.pt", os.path.basename(resume_path)
            )
            if checkpoint_match is not None:
                filename_iteration = int(checkpoint_match.group(1))
                if filename_iteration > runner.current_learning_iteration:
                    print(
                        f"Correcting checkpoint iteration metadata: {runner.current_learning_iteration} -> {filename_iteration}"
                    )
                    runner.current_learning_iteration = filename_iteration
                if hasattr(runner, "set_reference_policy"):
                    runner.set_reference_policy()
                    print("Frozen loaded actor as conservative continuation reference")
            return (runner, train_cfg)


task_registry = TaskRegistry()
