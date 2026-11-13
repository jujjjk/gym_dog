"""Flat-ground omnidirectional locomotion for the URDF4.0 quadruped (RS01 motors).

Signal path of one 20 ms policy step:

    actor output -> clip -> scale about the default pose
      -> target rate/acceleration limits            (50 Hz, real controller)
      -> closed-loop delay -> first-order response  (5 ms clock, identified RS01)
      -> PD on true joint state -> torque clip -> motor strength -> friction
                                                    (2.5 ms physics clock)

The actor sees only what the robot can measure: a disturbed, delayed IMU and
delayed, offset, noisy encoders.  Rewards and the critic use simulator truth.
"""

import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.rs01_go2_straight.rs01_actuator import (
    compute_rs01_joint_torques,
    limit_position_target,
    step_identified_position_response,
)

from .disturbances import DelayLine, ImuDisturbance, pulse_envelope, quat_rotate_inverse
from .gait_clock import LEG_PHASE_OFFSETS, CycleMean, leg_phase, scheduled_stance, swing_height_target

METRIC_NAMES = (
    "lin_vel_error_m_s",
    "yaw_rate_error_rad_s",
    "lin_vel_drift_m_s",
    "yaw_rate_drift_rad_s",
    "gait_contact_match",
    "motor_torque_over_continuous",
    "raw_torque_over_peak_fraction",
    "base_height_m",
)


class Urdf4OmniRobot(LeggedRobot):
    # ------------------------------------------------------------------ setup
    def _process_rigid_shape_props(self, props, env_id):
        dr = self.cfg.domain_rand
        if env_id == 0:
            friction = dr.friction_range if dr.randomize_friction else [self.cfg.terrain.static_friction] * 2
            restitution = dr.restitution_range if dr.randomize_restitution else [self.cfg.terrain.restitution] * 2
            self._shape_friction = np.random.uniform(friction[0], friction[1], self.num_envs)
            self._shape_restitution = np.random.uniform(restitution[0], restitution[1], self.num_envs)
        for shape in props:
            shape.friction = float(self._shape_friction[env_id])
            shape.restitution = float(self._shape_restitution[env_id])
        return props

    def _process_rigid_body_props(self, props, env_id):
        dr = self.cfg.domain_rand
        if env_id == 0:
            self._payload_kg = np.zeros(self.num_envs, dtype=np.float32)
            self._com_offset_m = np.zeros((self.num_envs, 3), dtype=np.float32)
        trunk = props[0]
        if dr.randomize_base_mass:
            self._payload_kg[env_id] = np.random.uniform(*dr.added_mass_range)
            trunk.mass += float(self._payload_kg[env_id])
        if dr.randomize_base_com:
            self._com_offset_m[env_id] = np.random.uniform(-1.0, 1.0, 3) * np.asarray(dr.base_com_offset_m)
            offset = self._com_offset_m[env_id]
            trunk.com = gymapi.Vec3(
                trunk.com.x + float(offset[0]), trunk.com.y + float(offset[1]), trunk.com.z + float(offset[2])
            )
        if dr.randomize_link_mass:
            for body in props[1:]:
                body.mass *= np.random.uniform(*dr.link_mass_scale_range)
        return props

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        dr = self.cfg.domain_rand
        if dr.randomize_armature:
            props["armature"][:] = np.random.uniform(*dr.armature_range_kg_m2, size=len(props))
        return props

    def _validate_asset(self):
        """Refuse to run if Isaac Gym's ordering differs from the configured contract."""
        configured = list(self.cfg.rs01_actuator.runtime_dof_order)
        if list(self.dof_names) != configured:
            raise ValueError(f"URDF4 DOF order changed: runtime={list(self.dof_names)}, configured={configured}")
        body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        if body_names[0] != "trunk":
            raise ValueError(f"Payload/push logic requires trunk as body 0, got {body_names[0]}")
        self.foot_legs = [body_names[int(index)].split("_")[0] for index in self.feet_indices.tolist()]
        if sorted(self.foot_legs) != ["FL", "FR", "RL", "RR"]:
            raise ValueError(f"Expected FL/FR/RL/RR foot bodies, got {self.foot_legs}")
        if len(self.termination_contact_indices) != 1 or len(self.penalised_contact_indices) != 12:
            raise ValueError("Expected 1 trunk and 12 hip/thigh/calf contact bodies")

    def _by_joint_type(self, values):
        result = []
        for name in self.dof_names:
            matches = [value for key, value in values.items() if f"_{key}_" in name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one joint-type value for {name}, got {matches}")
            result.append(float(matches[0]))
        return torch.tensor(result, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        # Sensor errors are produced by the imu / joint_sensor models instead.
        self.add_noise = bool(cfg.noise.add_noise)
        return torch.zeros_like(self.obs_buf[0])

    def _init_buffers(self):
        super()._init_buffers()
        self._validate_asset()
        cfg, n, device = self.cfg, self.num_envs, self.device
        if cfg.env.num_observations != cfg.env.frame_size * cfg.env.history_length:
            raise ValueError("num_observations must equal frame_size * history_length")
        if cfg.domain_rand.push_robots:
            raise ValueError("Use cfg.push (force pulses); the velocity-overwrite push must stay off")

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(n, -1, 13)
        self._refresh_feet()

        # Actuator contract.
        act = cfg.rs01_actuator
        self.physics_dt = float(self.sim_params.dt)
        ratio = act.response_update_dt_s / self.physics_dt
        self.response_substeps = int(round(ratio))
        if abs(self.dt - act.control_dt_s) > 1.0e-9:
            raise ValueError("Policy period must equal the RS01 control period")
        if self.response_substeps < 1 or abs(ratio - self.response_substeps) > 1.0e-6:
            raise ValueError("RS01 response period must be a whole number of physics steps")
        if cfg.control.decimation % self.response_substeps:
            raise ValueError("Policy step must align with the RS01 response clock")
        self.action_scale = self._by_joint_type(cfg.control.action_scale_by_joint)
        self.target_rate_limit = self._by_joint_type(act.target_rate_limit_rad_s)
        self.target_acceleration_limit = self._by_joint_type(act.target_acceleration_limit_rad_s2)
        self.nominal_delay_steps = int(round(act.closed_loop_delay_s / act.response_update_dt_s))
        delay_offsets = cfg.domain_rand.rs01_delay_step_offset_range
        if self.nominal_delay_steps + int(delay_offsets[0]) < 0:
            raise ValueError("RS01 delay randomization reaches a negative delay")
        default = self.default_dof_pos.repeat(n, 1)
        self.position_target = default.clone()
        self.target_rate = torch.zeros_like(default)
        self.response_target = default.clone()
        self.delay_buffer = default.unsqueeze(0).repeat(self.nominal_delay_steps + max(0, int(delay_offsets[1])) + 1, 1, 1)
        self.delay_steps = torch.full((n, self.num_dof), self.nominal_delay_steps, device=device, dtype=torch.long)
        self.response_gain = torch.full_like(default, act.response_gain)
        self.time_constant = torch.full_like(default, act.time_constant_s)
        self.coulomb_friction = torch.full_like(default, act.coulomb_friction_nm)
        self.motor_strength = torch.ones_like(default)
        self.kp = self.p_gains.repeat(n, 1)
        self.kd = self.d_gains.repeat(n, 1)
        self.peak_torque = torch.full_like(default, act.peak_torque_limit_nm)
        self.raw_pd_torques = torch.zeros_like(default)
        self.motor_torques = torch.zeros_like(default)
        self.raw_over_peak = torch.zeros(n, device=device)
        self._raw_over_peak_sum = torch.zeros(n, device=device)
        self.response_tick = 0

        # Sensors.
        longest = max(cfg.imu.latency_range_s[1], cfg.joint_sensor.latency_range_s[1])
        line_length = int(round(longest / self.physics_dt)) + 2
        self.imu_line = DelayLine(n, 7, line_length, device)  # body rate 3 + attitude quaternion 4
        self.joint_line = DelayLine(n, 2 * self.num_dof, line_length, device)
        self.imu_latency_steps = torch.zeros(n, device=device, dtype=torch.long)
        self.joint_latency_steps = torch.zeros(n, device=device, dtype=torch.long)
        self.imu = ImuDisturbance(n, device, self.dt, cfg.imu)
        self.encoder_offset = torch.zeros_like(default)
        self.obs_history = torch.zeros(n, cfg.env.history_length, cfg.env.frame_size, device=device)
        self._history_stale = torch.ones(n, device=device, dtype=torch.bool)

        # Gait clock, commands, curriculum.
        self.leg_offsets = torch.tensor([LEG_PHASE_OFFSETS[leg] for leg in self.foot_legs], device=device)
        self.clock_offset = torch.zeros(n, device=device)
        self.hip_dof_ids = [i for i, name in enumerate(self.dof_names) if "_hip_" in name]
        # Command minus body velocity (vx, vy, wz), averaged over one gait period.
        self.velocity_error = CycleMean(n, 3, round(cfg.gait.period_s / self.dt), device)
        self.velocity_drift = torch.zeros(n, 3, device=device)
        # The yaw part as the robot itself can measure it: command minus gyro.
        self.measured_yaw_error = CycleMean(n, 1, round(cfg.gait.period_s / self.dt), device)
        self.command_resample_steps = max(1, int(round(cfg.commands.resampling_time / self.dt)))
        fixed = cfg.commands.fixed_command
        self.command_override = (
            None if fixed is None else torch.tensor(fixed, device=device, dtype=torch.float).repeat(n, 1)
        )
        curriculum = cfg.command_curriculum
        self.command_level = float(curriculum.initial_level) if curriculum.enabled and not cfg.env.test else 1.0
        self._curriculum_scores = torch.zeros(3, device=device)  # linear, yaw, timeout share
        self._curriculum_last_step = 0

        # External force pulses.
        self.push_force_tensor = torch.zeros(n, self.num_bodies, 3, device=device)
        self.push_torque_tensor = torch.zeros_like(self.push_force_tensor)
        self.push_vector = torch.zeros(n, 3, device=device)
        self.push_force = torch.zeros(n, 3, device=device)
        self.push_elapsed = torch.ones(n, device=device)
        self.push_duration = torch.full((n,), 0.1, device=device)
        self.push_wait = torch.zeros(n, device=device)
        self.push_count = torch.zeros(n, device=device)

        # Creation-time randomization, exposed to the critic.
        self.friction_coeff = torch.tensor(self._shape_friction, device=device, dtype=torch.float).unsqueeze(1)
        self.payload_kg = torch.tensor(self._payload_kg, device=device).unsqueeze(1)
        self.com_offset_m = torch.tensor(self._com_offset_m, device=device)
        self.metric_sums = {name: torch.zeros(n, device=device) for name in METRIC_NAMES}
        # Steps actually simulated this episode.  episode_length_buf cannot be used:
        # the runner starts every robot at a random episode length.
        self.episode_steps = torch.zeros(n, device=device)

        everyone = torch.arange(n, device=device)
        self._randomize_episode(everyone)
        self._reset_sensors(everyone)
        self._clear_push(everyone)

    def _refresh_feet(self):
        # Advanced indexing copies, so re-slice after every tensor refresh.
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

    # ------------------------------------------------------- per-episode state
    def _uniform(self, shape, bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(*shape, device=self.device)

    def _randomize_episode(self, env_ids):
        """Resample controller, actuator and sensor parameters of the given robots."""
        count, cfg, dr, act = len(env_ids), self.cfg, self.cfg.domain_rand, self.cfg.rs01_actuator
        if count == 0:
            return
        training = not cfg.env.test
        shape = (count, self.num_dof)
        one = [1.0, 1.0]
        self.kp[env_ids] = self.p_gains * self._uniform(shape, dr.stiffness_scale_range if training and dr.randomize_gains else one)
        self.kd[env_ids] = self.d_gains * self._uniform(shape, dr.damping_scale_range if training and dr.randomize_gains else one)
        self.motor_strength[env_ids] = self._uniform(
            shape, dr.motor_strength_range if training and dr.randomize_motor_strength else one
        )
        randomize_actuator = training and dr.randomize_rs01_actuator
        self.response_gain[env_ids] = act.response_gain * self._uniform(
            shape, dr.rs01_response_gain_scale_range if randomize_actuator else one
        )
        self.time_constant[env_ids] = act.time_constant_s * self._uniform(
            shape, dr.rs01_time_constant_scale_range if randomize_actuator else one
        )
        self.coulomb_friction[env_ids] = act.coulomb_friction_nm * self._uniform(
            shape, dr.rs01_friction_scale_range if randomize_actuator else one
        )
        offsets = dr.rs01_delay_step_offset_range if randomize_actuator else [0, 0]
        self.delay_steps[env_ids] = self.nominal_delay_steps + torch.randint(
            int(offsets[0]), int(offsets[1]) + 1, shape, device=self.device
        )

        noisy = self.add_noise
        imu_latency = cfg.imu.latency_range_s if noisy and cfg.imu.enabled else [0.0, 0.0]
        joint_latency = cfg.joint_sensor.latency_range_s if noisy and cfg.joint_sensor.enabled else [0.0, 0.0]
        self.imu_latency_steps[env_ids] = torch.round(self._uniform((count,), imu_latency) / self.physics_dt).long()
        self.joint_latency_steps[env_ids] = torch.round(self._uniform((count,), joint_latency) / self.physics_dt).long()
        zero_offset = cfg.joint_sensor.zero_offset_rad if noisy and cfg.joint_sensor.enabled else 0.0
        self.encoder_offset[env_ids] = self._uniform(shape, [-zero_offset, zero_offset])
        self.imu.reset(env_ids)
        randomize_phase = training and cfg.gait.randomize_phase_on_reset
        self.clock_offset[env_ids] = torch.rand(count, device=self.device) if randomize_phase else 0.0

    def _sensor_truth(self):
        quat = self.root_states[:, 3:7]
        body_rate = quat_rotate_inverse(quat, self.root_states[:, 10:13])
        return torch.cat((body_rate, quat), dim=1), torch.cat((self.dof_pos, self.dof_vel), dim=1)

    def _reset_sensors(self, env_ids):
        imu_sample, joint_sample = self._sensor_truth()
        self.imu_line.reset(env_ids, imu_sample[env_ids])
        self.joint_line.reset(env_ids, joint_sample[env_ids])
        self.measured_yaw_error.reset(env_ids)
        self._history_stale[env_ids] = True

    def _clear_push(self, env_ids):
        self.push_elapsed[env_ids] = 1.0
        self.push_duration[env_ids] = 0.1
        self.push_wait[env_ids] = self._uniform((len(env_ids),), self.cfg.push.first_delay_s)
        self.push_vector[env_ids] = 0.0
        self.push_force[env_ids] = 0.0
        self.push_count[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        noise = 0.0 if self.cfg.env.test else float(self.cfg.init_state.reset_dof_position_noise_rad)
        position = self.default_dof_pos + self._uniform((len(env_ids), self.num_dof), [-noise, noise])
        self.dof_pos[env_ids] = torch.max(torch.min(position, self.dof_pos_limits[:, 1]), self.dof_pos_limits[:, 0])
        self.dof_vel[env_ids] = 0.0
        # The controller restarts holding the default pose.
        self.position_target[env_ids] = self.default_dof_pos
        self.target_rate[env_ids] = 0.0
        self.response_target[env_ids] = self.default_dof_pos
        self.delay_buffer[:, env_ids] = self.default_dof_pos
        self.raw_pd_torques[env_ids] = 0.0
        self.motor_torques[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        count, init = len(env_ids), self.cfg.init_state
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if not self.cfg.env.test:
            if init.randomize_yaw:
                yaw = self._uniform((count,), [-math.pi, math.pi])
                self.root_states[env_ids, 3:5] = 0.0
                self.root_states[env_ids, 5] = torch.sin(0.5 * yaw)
                self.root_states[env_ids, 6] = torch.cos(0.5 * yaw)
            linear, angular = init.reset_linear_velocity_m_s, init.reset_angular_velocity_rad_s
            self.root_states[env_ids, 7:9] = self._uniform((count, 2), [-linear, linear])
            self.root_states[env_ids, 10:13] = self._uniform((count, 3), [-angular, angular])
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        steps = self.episode_steps[env_ids].clamp(min=1.0)
        self._update_command_curriculum(env_ids, steps)
        metrics = {name: (total[env_ids] / steps).mean() for name, total in self.metric_sums.items()}
        pushes = self.push_count[env_ids].mean()
        super().reset_idx(env_ids)
        self._randomize_episode(env_ids)
        self._reset_sensors(env_ids)
        self._clear_push(env_ids)
        self.velocity_error.reset(env_ids)
        self.velocity_drift[env_ids] = 0.0
        for total in self.metric_sums.values():
            total[env_ids] = 0.0
        self.episode_steps[env_ids] = 0.0
        self.extras["episode"].update(metrics)
        self.extras["episode"]["command_level"] = self.command_level
        self.extras["episode"]["push_strength"] = self._push_strength()
        self.extras["episode"]["pushes_per_episode"] = pushes

    # ---------------------------------------------------------------- commands
    def _resample_commands(self, env_ids):
        count = len(env_ids)
        if count == 0:
            return
        cfg = self.cfg.commands
        if self.command_override is not None:
            self.commands[env_ids, :3] = self.command_override[env_ids]
            return
        level, ranges = self.command_level, self.command_ranges
        command = torch.stack(
            (
                self._uniform((count,), [level * ranges["lin_vel_x"][0], level * ranges["lin_vel_x"][1]]),
                self._uniform((count,), [level * ranges["lin_vel_y"][0], level * ranges["lin_vel_y"][1]]),
                self._uniform((count,), [level * ranges["ang_vel_yaw"][0], level * ranges["ang_vel_yaw"][1]]),
            ),
            dim=1,
        )
        single_axis = torch.rand(count, device=self.device) < cfg.single_axis_probability
        kept_axis = torch.randint(0, 3, (count,), device=self.device)
        keep = torch.nn.functional.one_hot(kept_axis, 3).bool() | ~single_axis.unsqueeze(1)
        command = command * keep
        too_small = (command[:, :2].norm(dim=1) < cfg.min_planar_speed_m_s) & (
            command[:, 2].abs() < cfg.min_yaw_speed_rad_s
        )
        stand = too_small | (torch.rand(count, device=self.device) < cfg.stand_probability)
        self.commands[env_ids, :3] = command * (~stand).unsqueeze(1)

    def set_command_override(self, commands):
        """Playback/evaluation: hold [num_envs, 3] (vx, vy, wz) instead of sampling."""
        self.command_override = commands.to(self.device, dtype=torch.float).reshape(self.num_envs, 3).clone()
        self.commands[:, :3] = self.command_override

    def _moving(self):
        return (self.commands[:, :2].norm(dim=1) > 1.0e-4) | (self.commands[:, 2].abs() > 1.0e-4)

    def _update_command_curriculum(self, env_ids, steps):
        c = self.cfg.command_curriculum
        if not c.enabled or self.cfg.env.test or self.command_level >= 1.0:
            return
        # reward_scales already include dt, so this is the episode-mean reward value in [0, 1].
        batch = torch.stack(
            (
                (self.episode_sums["tracking_lin_vel"][env_ids] / (steps * self.reward_scales["tracking_lin_vel"])).mean(),
                (self.episode_sums["tracking_ang_vel"][env_ids] / (steps * self.reward_scales["tracking_ang_vel"])).mean(),
                self.time_out_buf[env_ids].float().mean(),
            )
        )
        self._curriculum_scores += c.ema_rate * (batch - self._curriculum_scores)
        # One full episode must pass at the new level before judging it.
        if self.common_step_counter - self._curriculum_last_step < self.max_episode_length:
            return
        linear, yaw, timeout = self._curriculum_scores.tolist()
        if linear > c.min_linear_tracking and yaw > c.min_yaw_tracking and timeout > c.min_timeout_fraction:
            self.command_level = min(1.0, self.command_level + c.level_step)
            self._curriculum_last_step = self.common_step_counter

    # --------------------------------------------------------------- gait clock
    def _clock_phase(self):
        elapsed = self.episode_length_buf.float() * self.dt
        return torch.remainder(elapsed / self.cfg.gait.period_s + self.clock_offset, 1.0)

    def _leg_phase(self):
        return leg_phase(self._clock_phase(), self.leg_offsets)

    def _scheduled_stance(self):
        """All four feet are scheduled down while standing still."""
        stance = scheduled_stance(self._leg_phase(), self.cfg.gait.stance_ratio)
        return stance | ~self._moving().unsqueeze(1)

    # --------------------------------------------------------------- simulation
    def step(self, actions):
        clip = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip, clip).to(self.device)
        self.position_target, self.target_rate = limit_position_target(
            self.default_dof_pos + self.action_scale * self.actions,
            self.position_target,
            self.target_rate,
            self.target_rate_limit,
            self.target_acceleration_limit,
            self.dt,
        )
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self._apply_push()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            imu_sample, joint_sample = self._sensor_truth()
            self.imu_line.push(imu_sample)
            self.joint_line.push(joint_sample)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        if self.response_tick % self.response_substeps == 0:
            self.delay_buffer = torch.roll(self.delay_buffer, shifts=1, dims=0)
            self.delay_buffer[0] = self.position_target
            delayed_target = torch.gather(
                self.delay_buffer.permute(1, 2, 0), dim=2, index=self.delay_steps.unsqueeze(-1)
            ).squeeze(-1)
            self.response_target = step_identified_position_response(
                self.response_target,
                delayed_target,
                self.default_dof_pos,
                self.response_gain,
                self.time_constant,
                self.cfg.rs01_actuator.response_update_dt_s,
            )
        self.response_tick += 1
        self.raw_pd_torques, self.motor_torques, applied = compute_rs01_joint_torques(
            self.response_target,
            self.dof_pos,
            self.dof_vel,
            self.kp,
            self.kd,
            self.peak_torque,
            self.coulomb_friction,
            self.cfg.rs01_actuator.friction_smoothing_rad_s,
        )
        excess = torch.clamp(self.raw_pd_torques.abs() - self.peak_torque, min=0.0) / self.peak_torque
        self._raw_over_peak_sum += excess.square().mean(dim=1)
        # Torque-constant error scales the delivered torque, not the friction loss.
        return applied + (self.motor_strength - 1.0) * self.motor_torques

    def _push_strength(self):
        c = self.cfg.push
        blend = 1.0 if c.ramp_s <= 0.0 else min(1.0, self.common_step_counter * self.dt / c.ramp_s)
        return c.initial_strength + (1.0 - c.initial_strength) * blend

    def _apply_push(self):
        c = self.cfg.push
        if not c.enabled:
            return
        dt = self.physics_dt
        self.push_wait -= dt
        start = (self.push_wait <= 0.0).nonzero().flatten()
        if len(start):
            angle = self._uniform((len(start),), [0.0, 2.0 * math.pi])
            peak = self._push_strength() * self._uniform((len(start),), c.peak_force_n)
            self.push_vector[start, 0] = peak * torch.cos(angle)
            self.push_vector[start, 1] = peak * torch.sin(angle)
            self.push_duration[start] = self._uniform((len(start),), c.duration_s)
            self.push_elapsed[start] = 0.0
            self.push_wait[start] = self.push_duration[start] + self._uniform((len(start),), c.interval_s)
            self.push_count[start] += 1.0
        envelope = pulse_envelope(self.push_elapsed + 0.5 * dt, self.push_duration)
        self.push_force = self.push_vector * envelope.unsqueeze(1)
        self.push_elapsed += dt
        if not bool((envelope > 0.0).any()):
            return  # Applied forces last one physics step, so nothing to clear.
        self.push_force_tensor[:, 0] = self.push_force
        # Force applied above the CoM: torque = r x F with r = (0, 0, height).
        self.push_torque_tensor[:, 0, 0] = -c.application_height_m * self.push_force[:, 1]
        self.push_torque_tensor[:, 0, 1] = c.application_height_m * self.push_force[:, 0]
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.push_force_tensor),
            gymtorch.unwrap_tensor(self.push_torque_tensor),
            gymapi.ENV_SPACE,
        )

    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._refresh_feet()
        resample = (self.episode_length_buf % self.command_resample_steps == 0).nonzero().flatten()
        self._resample_commands(resample)
        error = self.commands[:, :3] - torch.cat((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]), dim=1)
        self.velocity_error.push(error)
        self.velocity_drift = self.velocity_error.mean()
        self.imu.step()
        self.raw_over_peak = self._raw_over_peak_sum / self.cfg.control.decimation
        self._raw_over_peak_sum = torch.zeros_like(self._raw_over_peak_sum)

    def check_termination(self):
        trunk_force = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        fallen = torch.any(trunk_force > 1.0, dim=1)
        fallen |= self.projected_gravity[:, 2] > -0.6  # Tilted more than ~53 deg.
        fallen |= self.root_states[:, 2] < self.cfg.rewards.min_base_height_m
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = fallen | self.time_out_buf

    def compute_reward(self):
        super().compute_reward()
        contact = self.get_foot_contact_mask()
        samples = {
            "lin_vel_error_m_s": (self.commands[:, :2] - self.base_lin_vel[:, :2]).norm(dim=1),
            "yaw_rate_error_rad_s": (self.commands[:, 2] - self.base_ang_vel[:, 2]).abs(),
            "lin_vel_drift_m_s": self.velocity_drift[:, :2].norm(dim=1),
            "yaw_rate_drift_rad_s": self.velocity_drift[:, 2].abs(),
            "gait_contact_match": (contact == self._scheduled_stance()).float().mean(dim=1),
            "motor_torque_over_continuous": (
                self.motor_torques.abs() > self.cfg.rs01_actuator.continuous_torque_nm
            ).float().mean(dim=1),
            "raw_torque_over_peak_fraction": (self.raw_pd_torques.abs() > self.peak_torque).float().mean(dim=1),
            "base_height_m": self.root_states[:, 2],
        }
        for name, value in samples.items():
            self.metric_sums[name] += value
        self.episode_steps += 1.0

    # ------------------------------------------------------------- observations
    def _frame(self, gyro, gravity, joint_position, joint_velocity, yaw_error_mean):
        scales = self.obs_scales
        angle = 2.0 * math.pi * self._clock_phase()
        return torch.cat(
            (
                gyro * scales.ang_vel,
                gravity,
                self.commands[:, :3] * self.commands_scale,
                (joint_position - self.default_dof_pos) * scales.dof_pos,
                joint_velocity * scales.dof_vel,
                self.actions,
                torch.stack((torch.sin(angle), torch.cos(angle)), dim=1),
                # Stride-averaged yaw-rate error.  Within a stride the yaw rate swings
                # by ~0.2 rad/s, hiding the ~0.05 rad/s offset that bends the path.
                yaw_error_mean * scales.yaw_rate_error_mean,
            ),
            dim=1,
        )

    def compute_observations(self):
        # Root and joint tensors are already current for robots reset this step.
        quat = self.root_states[:, 3:7]
        true_gyro = quat_rotate_inverse(quat, self.root_states[:, 10:13])
        true_gravity = quat_rotate_inverse(quat, self.gravity_vec)
        if self.add_noise:
            sensor = self.cfg.joint_sensor
            imu_sample = self.imu_line.read(self.imu_latency_steps)
            gyro, gravity = self.imu.measure(imu_sample[:, 3:7], imu_sample[:, :3])
            joint_sample = self.joint_line.read(self.joint_latency_steps)
            position_noise = sensor.position_noise_rad if sensor.enabled else 0.0
            velocity_noise = sensor.velocity_noise_rad_s if sensor.enabled else 0.0
            shape = (self.num_envs, self.num_dof)
            joint_position = (
                joint_sample[:, : self.num_dof] + self.encoder_offset
                + self._uniform(shape, [-position_noise, position_noise])
            )
            joint_velocity = joint_sample[:, self.num_dof :] + self._uniform(shape, [-velocity_noise, velocity_noise])
        else:
            gyro, gravity, joint_position, joint_velocity = true_gyro, true_gravity, self.dof_pos, self.dof_vel

        self.measured_yaw_error.push(self.commands[:, 2:3] - gyro[:, 2:3])
        frame = self._frame(gyro, gravity, joint_position, joint_velocity, self.measured_yaw_error.mean())
        self.obs_history = torch.roll(self.obs_history, shifts=1, dims=1)
        self.obs_history[:, 0] = frame
        # A new episode has no past: fill its whole history with the first frame.
        self.obs_history[self._history_stale] = frame[self._history_stale].unsqueeze(1)
        self._history_stale[:] = False
        self.obs_buf = self.obs_history.reshape(self.num_envs, -1)

        base_velocity = quat_rotate_inverse(quat, self.root_states[:, 7:10])
        self.privileged_obs_buf = torch.cat(
            (
                self._frame(true_gyro, true_gravity, self.dof_pos, self.dof_vel, self.velocity_drift[:, 2:3]),
                base_velocity * self.obs_scales.lin_vel,
                (self.root_states[:, 2:3] - self.cfg.rewards.base_height_target) * 5.0,
                self.get_foot_contact_mask().float(),
                self._scheduled_stance().float(),
                self.feet_pos[:, :, 2] * 10.0,
                self.push_force / 20.0,
                self.friction_coeff,
                self.payload_kg,
                self.com_offset_m * 10.0,
                self.torques * 0.05,
                (self.motor_strength - 1.0) * 5.0,
            ),
            dim=1,
        )

    # ------------------------------------------------------------------ rewards
    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.rewards.yaw_tracking_sigma)

    def _reward_cycle_lin_vel(self):
        """Stride-averaged planar velocity error: the part that shifts the path sideways.

        exp(-|e|/sigma) keeps a constant pull all the way to zero, where the
        per-step Gaussian tracking kernel is flat.
        """
        return torch.exp(-self.velocity_drift[:, :2].norm(dim=1) / self.cfg.rewards.cycle_lin_vel_sigma_m_s)

    def _reward_cycle_ang_vel(self):
        """Stride-averaged yaw-rate error: the part that bends a straight path."""
        return torch.exp(-self.velocity_drift[:, 2].abs() / self.cfg.rewards.cycle_yaw_rate_sigma_rad_s)

    def _reward_gait_contact(self):
        """Share of feet whose contact state matches the diagonal-trot schedule."""
        return (self.get_foot_contact_mask() == self._scheduled_stance()).float().mean(dim=1)

    def _reward_swing_clearance(self):
        """Shortfall below the scheduled swing height; lifting higher is not charged here."""
        gait = self.cfg.gait
        phase = self._leg_phase()
        swing = ~self._scheduled_stance()
        target = swing_height_target(phase, gait.stance_ratio, gait.foot_radius_m, gait.swing_clearance_m)
        shortfall = torch.clamp(target - self.feet_pos[:, :, 2], min=0.0) / gait.swing_clearance_m
        return (shortfall.square() * swing).sum(dim=1) / swing.sum(dim=1).clamp(min=1)

    def _reward_feet_slip(self):
        contact = self.get_foot_contact_mask().float()
        return (self.feet_vel[:, :, :2].square().sum(dim=2) * contact).sum(dim=1)

    def _reward_hip_deviation(self):
        hip_error = self.dof_pos[:, self.hip_dof_ids] - self.default_dof_pos[:, self.hip_dof_ids]
        return hip_error.square().sum(dim=1)

    def _reward_stand_still(self):
        posture = (self.dof_pos - self.default_dof_pos).abs().sum(dim=1)
        motion = 0.05 * self.dof_vel.abs().sum(dim=1)
        return (posture + motion) * (~self._moving()).float()

    def _reward_raw_torque_over_peak(self):
        """PD requests beyond the torque cap, averaged over the physics substeps."""
        return self.raw_over_peak
