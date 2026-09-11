"""Explicit user-run single motion trial; never run by deployment/build."""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vx', type=float, default=0.)
    parser.add_argument('--vy', type=float, default=0.)
    parser.add_argument('--wz', type=float, default=0.)
    parser.add_argument('--seconds', type=float, default=3.)
    opts = parser.parse_args(args)
    values = (opts.vx, opts.vy, opts.wz)
    if (not all(math.isfinite(v) for v in (*values, opts.seconds)) or
            not 0 < opts.seconds <= 6. or
            any(abs(v) > cap for v, cap in zip(values, (.30, .20, .30)))):
        parser.error('Limits: |vx|<=0.30, |vy|<=0.20, |wz|<=0.30, 0<seconds<=6')
    rclpy.init()
    node = rclpy.create_node('model6850_single_trial')
    pub = node.create_publisher(Twist, '/mydog/model6850/cmd_vel', 1)
    client = node.create_client(SetBool, '/mydog/model6850/arm')
    status = {}
    status_time = [0.]

    def on_status(msg):
        try:
            status.clear()
            status.update(json.loads(msg.data))
            status_time[0] = time.monotonic()
        except (ValueError, TypeError):
            status.clear()

    sub = node.create_subscription(String, '/mydog/model6850/status', on_status, 1)

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
        if not any(values):
            set_arm(False)
            return
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if (status.get('mode') == 'ready' and status.get('walk_start_stable')
                    and status.get('send') and not status.get('stand_only')
                    and time.monotonic() - status_time[0] < .2
                    and pub.get_subscription_count() > 0):
                break
        else:
            raise RuntimeError('Not ready: require fresh ready, stable and send=true, stand_only=false')
        set_arm(True)
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = values
        # Duration includes the one-second stable start gate.
        deadline = time.monotonic() + opts.seconds
        while time.monotonic() < deadline:
            if (time.monotonic() - status_time[0] > .25 or
                    status.get('mode') in ('fault', 'soft_hold')):
                raise RuntimeError('Trial aborted: stale status or controller fault/inhibit')
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
