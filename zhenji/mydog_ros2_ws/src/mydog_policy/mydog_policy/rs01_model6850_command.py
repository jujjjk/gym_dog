"""Explicit user-run single motion trial; never run by deployment/build."""

import argparse
import json
import math
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool


def main(args=None, *, speed_caps=(.30, .20, .30), allow_continuous=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vx', type=float, default=0.)
    parser.add_argument('--vy', type=float, default=0.)
    parser.add_argument('--wz', type=float, default=0.)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--seconds', type=float, default=3.)
    duration.add_argument('--continuous', action='store_true', help='Run until Ctrl+C; requires a continuous-enabled B23500 controller')
    parser.add_argument('--namespace', choices=['/mydog/model6850','/mydog/model18000','/mydog/model23500'], default='/mydog/model6850')
    parser.add_argument('--march', action='store_true', help='Explicit gait-on zero-speed trial')
    opts = parser.parse_args(args)
    if opts.continuous and not allow_continuous:
        parser.error('Continuous commands are only available for B23500')
    values = (opts.vx, opts.vy, opts.wz)
    if (not all(math.isfinite(v) for v in (*values, opts.seconds)) or
            not 0 < opts.seconds <= 6. or
            any(abs(v) > cap for v, cap in zip(values, speed_caps))):
        parser.error(f'Limits: |vx|<={speed_caps[0]}, |vy|<={speed_caps[1]}, |wz|<={speed_caps[2]}, 0<seconds<=6')
    # Keep the ROS context alive through Ctrl+C so finally can disarm.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('model6850_single_trial')
    pub = node.create_publisher(Twist, opts.namespace + '/cmd_vel', 1)
    client = node.create_client(SetBool, opts.namespace + '/arm')
    status = {}
    status_time = [0.]

    def on_status(msg):
        try:
            status.clear()
            status.update(json.loads(msg.data))
            status_time[0] = time.monotonic()
        except (ValueError, TypeError):
            status.clear()

    sub = node.create_subscription(String, opts.namespace + '/status', on_status, 1)

    def set_arm(value):
        req = SetBool.Request()
        req.data = value
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
        if not future.done() or future.result() is None:
            raise RuntimeError('Arm service timed out')
        response = future.result()
        if not response.success:
            raise RuntimeError(response.message)
        print(response.message, flush=True)

    try:
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError('A6850 hardware node /arm service unavailable')
        if not any(values) and not opts.march:
            set_arm(False)
            return
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if (status.get('mode') == 'ready' and status.get('walk_start_stable')
                    and status.get('send') and not status.get('stand_only')
                    and status.get('timing_ready', True)
                    and status.get('observation_temporal_ok', True)
                    and status.get('imu_calibrated', True)
                    and time.monotonic() - status_time[0] < .2
                    and pub.get_subscription_count() > 0):
                break
        else:
            raise RuntimeError('Not ready: require fresh ready/stable, timing, observation freshness, calibration and send=true; status=' + str({k:status.get(k) for k in ('mode','walk_start_stable','timing_ready','observation_temporal_ok','imu_calibrated','aligned_motor_age_ms','aligned_imu_age_ms')}))
        if opts.continuous and not status.get('continuous_commands', False):
            raise RuntimeError('Restart with continuous_commands=true before using --continuous')
        set_arm(True)
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = values
        # Duration includes the one-second stable start gate.
        deadline = None if opts.continuous else time.monotonic() + opts.seconds
        while deadline is None or time.monotonic() < deadline:
            status_age = time.monotonic() - status_time[0]
            if status_age > .25:
                raise RuntimeError(f'Trial aborted: controller status stale ({status_age:.3f}s > 0.250s)')
            if status.get('mode') in ('fault', 'soft_hold'):
                reason = status.get('walk_inhibit_reason') if status.get('mode') == 'soft_hold' else status.get('last_fault_reason')
                raise RuntimeError('Trial aborted: ' + str(status.get('mode')) + ': ' +
                                   str(reason or status.get('trial_reason') or 'reason unavailable'))
            pub.publish(message)
            rclpy.spin_once(node, timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        # Redundant zero + disarm; a killed client is covered by node timeout.
        if rclpy.ok():
            try:
                try:
                    if client.service_is_ready():
                        set_arm(False)
                finally:
                    for _ in range(3):
                        pub.publish(Twist())
                        rclpy.spin_once(node, timeout_sec=.05)
            finally:
                node.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    main()
