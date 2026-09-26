"""B18000 cycle-coherent capture. Defaults remain dry-run; no automatic arming."""
import json
import time
import numpy as np
import rclpy
from .rs01_model18000_node import Rs01Model18000Node
from .capture61 import CaptureWriter

class Capture61Node(Rs01Model18000Node):
    capture_schema = 'b18000_capture61_v1'
    def __init__(self):
        self.capture=None;self.capture_step=0;self.capture_imu=None
        self.wall_minus_mono=time.time()-time.monotonic()
        super().__init__()
        self.declare_parameter('capture_dir','')
        path=self.get_parameter('capture_dir').value
        if not path:raise ValueError('capture_dir is required and must be a new session directory')
        metadata=dict(schema=self.capture_schema,created_wall=time.time(),
            wall_minus_monotonic=self.wall_minus_mono,
            timestamp_basis='policy: host monotonic; imu/motor: host wall snapshot converted with startup offset, NOT device acquisition',
            acquisition_sync_verified=False,units='angles rad; rates rad/s; velocities m/s; torque Nm; temperature C',
            real_joint_order=['FR_hip','FR_thigh','FR_calf','FL_hip','FL_thigh','FL_calf','RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf'],
            policy_joint_order=list(self.contract.joint_names),odom_leg_order=['FL','FR','RL','RR'],
            imu_rotation_base_from_imu=np.asarray(self.imu.R_BASE_IMU).tolist(),
            missing=['independent device IMU acquisition timestamp','raw 100Hz IMU frame stream','motor current','external ground truth','stance score'],
            target_semantics='target_real is the control-cycle queued target after protection, NOT acknowledgement of physical execution',
            observation_semantics='exact controller observation; policy_evaluated=false means no policy ran (stand/startup placeholders)',
            raw_action_semantics='unclipped ONNX output only when policy_evaluated=true',
            contract=self.contract.raw)
        pipeline = getattr(self, 'observation_pipeline', None)
        if pipeline is not None:
            metadata['imu_rotation_base_from_imu'] = pipeline.rotation.tolist()
            metadata['observation_pipeline'] = dict(
                time_basis='host_reception_and_board_age_estimate', acquisition_sync_verified=False,
                motor_acquisition_timestamp=None, imu_acquisition_timestamp=None,
                alignment='latest IMU in reception mode; nearest history only in strict_host_alignment',
                observation_timing_mode=pipeline.timing_mode,
                reception_age_limit_ms=pipeline.max_age_ms, imu_internal_receive_span_limit_ms=60.,
                skew_is_hard_gate=pipeline.timing_mode=='strict_host_alignment',
                age_limit_ms=pipeline.max_age_ms, skew_limit_ms=10., quality_bad_hold_sec=.20,
                filters='preview only; not applied to policy or PD',
                gyro_bias='base frame; calibrated separately after mounting rotation',
                quaternion_frame='raw IMU to world', rpy_frame='base to world',
                euler_source='quaternion', required_imu_frames=['RAW','QUAT'])
            alignment = getattr(self, 'common_time_alignment', None)
            if alignment is not None:
                metadata['observation_pipeline'].update(
                    time_basis='mapped_mcu_can_reception_and_serial_reception_monotonic',
                    observation_timing_mode='common_time', alignment='50ms delayed linear q/dq/gyro and quaternion SLERP; real brackets required',
                    delay_ms=alignment.delay*1000., extrapolation=False, pd_feedback='latest; not delayed',
                    timestamp_accuracy='transport estimate; interpolation skew is not acquisition accuracy')
        self.capture=CaptureWriter(path,metadata)

    def _fresh_state(self):
        state=super()._fresh_state()
        self.capture_imu=state[1]
        return state

    def _publish(self,observation,action,target_real,odometry,roll,pitch,yaw,motor,torque_info,now,guard_odometry):
        if self.capture is not None and self.capture_imu is not None:
            imu=self.capture_imu;actor=self.core.actor
            evaluated=self.mode=='walk' and actor.last_capture is not None
            diag=actor.last_capture if evaluated else {}
            nan=float('nan');obs=np.asarray(observation)
            imu_t=float(imu.stamp)-self.wall_minus_mono
            motor_t=float(motor.stamp)-self.wall_minus_mono
            command=np.asarray(actor.command if evaluated else self._command_vector())
            vel=np.asarray(diag.get('estimated_velocity',odometry['base_linear_velocity']))
            effective=obs[9:12]/self.contract.command_scale if evaluated else [nan]*3
            row=dict(timestamp_policy=now,timestamp_imu=imu_t,timestamp_motor=motor_t,
                timestamp_imu_host_wall=float(imu.stamp),timestamp_motor_host_wall=float(motor.stamp),
                control_step=self.capture_step,mode=self.mode,policy_evaluated=evaluated,
                enable_send=self.enable_send,imu_calibrated=self.imu_calibrated,
                imu_age_ms=(now-imu_t)*1000,motor_age_ms=(now-motor_t)*1000,
                sensor_skew_ms=abs(imu_t-motor_t)*1000,acquisition_sync_verified=False,
                roll=roll,pitch=pitch,yaw=yaw,heading_target=float(self.core.heading_target),
                heading_error=float(np.arctan2(np.sin(self.core.heading_target-yaw),np.cos(self.core.heading_target-yaw))),
                odom_confidence=float(obs[60]) if evaluated else float(odometry['confidence']),
                guard_odom_confidence=float(guard_odometry['confidence']),
                heading_healthy=bool(self.heading_consistency_state.get('healthy')),
                phase=diag.get('phase',nan),gait_enable=1 if evaluated else 0,
                imu_valid=bool(imu.valid),motor_valid=bool(motor.valid),
                capture_dropped=self.capture.dropped,capture_error=self.capture.error)
            def vec(prefix,values,names):
                values=np.asarray(values).reshape(-1)
                for k,v in zip(names,values):row[prefix+k]=float(v)
            axes=['x','y','z'];joints=['FR_hip','FR_thigh','FR_calf','FL_hip','FL_thigh','FL_calf','RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']
            vec('cmd_',command,['vx','vy','wz']);vec('effective_target_',effective,['vx','vy','wz'])
            vec('est_v',vel,axes);vec('gyro_',self.corrected_gyro_rad_s,axes)
            vec('gyro_uncalibrated_base_',imu.gyro_rad_s,axes)
            pipeline = getattr(self, 'observation_pipeline', None)
            sensor_gyro = pipeline.diagnostics['raw_gyro'] if pipeline is not None else np.asarray(self.imu.R_BASE_IMU).T@imu.gyro_rad_s
            vec('gyro_sensor_',sensor_gyro,axes)
            vec('gyro_bias_',self.gyro_bias_rad_s,axes)
            vec('projected_gravity_',imu.projected_gravity,axes)
            vec('acc_sensor_g_',imu.acc_g,axes);vec('mag_sensor_uT_',imu.mag_uT,axes)
            vec('quat_',imu.quat_wxyz,['w','x','y','z'])
            vec('q_',motor.q_real,joints);vec('dq_',motor.dq_real,joints)
            alignment = getattr(self, 'common_time_alignment', None)
            if alignment is not None:
                aligned = alignment.sample or {}
                vec('aligned_q_', aligned.get('q_real', [nan]*12), joints)
                vec('aligned_dq_', aligned.get('dq_real', [nan]*12), joints)
                vec('aligned_gyro_uncalibrated_', aligned.get('gyro', [nan]*3), axes)
                vec('aligned_gravity_', aligned.get('gravity', [nan]*3), axes)
                row['aligned_yaw_rad'] = aligned.get('yaw', nan)
            vec('target_q_',target_real,joints)
            for prefix,attr in [('motor_torque_','torque'),('motor_temperature_','temp'),('motor_error_','error_code'),('motor_age_ms_','age_ms'),('motor_board_tick_','board_tick_ms'),('motor_snapshot_seq_','snapshot_seq'),('motor_last_update_ts_','last_update_ts')]:
                vec(prefix,getattr(motor,attr),joints)
            vec('obs_',obs,[f'{i:02d}' for i in range(61)])
            vec('raw_action_',diag.get('raw_action',[nan]*12),[f'{i+1:02d}' for i in range(12)])
            vec('action_',action,[f'{i+1:02d}' for i in range(12)])
            vec('target_rate_',diag.get('target_rate',[nan]*12),list(self.contract.joint_names))
            for name in ['raw_pd_torque_nm','safe_pd_torque_nm']:
                vec(name+'_',torque_info[name],list(self.contract.joint_names))
            vec('next_active_torque_limit_',self.active_limit_policy,list(self.contract.joint_names))
            legs=['FL','FR','RL','RR'];vec('stance_',odometry['stance_mask'],legs)
            for name in ['foot_position','foot_velocity','velocity_by_foot']:
                vec(name+'_',odometry[name],[leg+'_'+axis for leg in legs for axis in axes])
            row['estimated_vx_minus_target']=float(vel[0]-effective[0]);row['estimated_vy_minus_target']=float(vel[1]-effective[1])
            for prefix, od in [('guard_', guard_odometry), ('actor_', diag.get('odometry') or {})]:
                row[prefix+'selected_pair_index'] = int(od.get('selected_pair_index', -1))
                row[prefix+'legal_diagonal_support'] = bool(od.get('legal_diagonal_support', False))
                row[prefix+'pair_residual_m_s'] = float(od.get('pair_residual_m_s', nan))
                row[prefix+'confidence'] = float(od.get('confidence', nan))
                vec(prefix+'raw_velocity_', od.get('raw_base_velocity', [nan]*3), axes)
                vec(prefix+'filtered_velocity_', od.get('base_linear_velocity', [nan]*3), axes)
                vf = np.asarray(od.get('velocity_by_foot', np.full((4,3), nan)))
                vec(prefix+'velocity_by_foot_', vf, [leg+'_'+axis for leg in legs for axis in axes])
                for pair_index, pair in enumerate(((0,3), (1,2))):
                    row[prefix+'candidate_pair_residual_'+str(pair_index)] = float(np.linalg.norm((vf[pair[0]]-vf[pair[1]])[:2]))
            if pipeline is not None:
                row.update(pipeline.scalar_diagnostics())
                for name in ('raw_dq','preview_filtered_dq'):
                    vec(name+'_', pipeline.diagnostics[name], joints)
                for name in ('raw_gyro','base_gyro','preview_filtered_gyro','raw_gravity','base_gravity'):
                    vec(name+'_', pipeline.diagnostics[name], axes)
            self.capture.submit(row);self.capture_step+=1
        return super()._publish(observation,action,target_real,odometry,roll,pitch,yaw,motor,torque_info,now,guard_odometry)

    def _extra_status(self):
        result=super()._extra_status()
        if self.capture is not None:
            result.update(capture_rows=self.capture.written,capture_dropped=self.capture.dropped,
                          capture_error=self.capture.error,capture_path=str(self.capture.path))
        return result

    def destroy_node(self):
        try:return super().destroy_node()
        finally:
            if self.capture is not None:self.capture.close()

def main(args=None):
    rclpy.init(args=args);node=None
    try:node=Capture61Node();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
