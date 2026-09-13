"""Isaac-matching A6850 omni sequence. Operator-run only; not started by launch."""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool

# Same matrix as the Isaac `play_rs01_go2_omni` demo: 5 s step-in-place
# between each move, gait stays on for the whole playback.
MOVES = (
    ("前进", 0.20, 0.00, 0.00),
    ("后退", -0.20, 0.00, 0.00),
    ("左移", 0.00, 0.20, 0.00),
    ("右移", 0.00, -0.20, 0.00),
    ("左转", 0.00, 0.00, 0.30),
    ("右转", 0.00, 0.00, -0.30),
    ("组合", 0.30, 0.10, 0.25),
    ("反向组合", -0.30, -0.10, -0.25),
)


def isaac_sequence():
    sequence = [("踏步", 0.00, 0.00, 0.00)]
    for move in MOVES:
        sequence.extend([move, ("踏步", 0.00, 0.00, 0.00)])
    return sequence


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment-sec', type=float, default=5.)
    opts = parser.parse_args(args)
    if not math.isfinite(opts.segment_sec) or not 0 < opts.segment_sec <= 5.:
        parser.error('segment-sec must be in (0, 5]')

    sequence = isaac_sequence()
    rclpy.init()
    node = rclpy.create_node('model6850_omni_sequence')
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

    node.create_subscription(String, '/mydog/model6850/status', on_status, 1)

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

    def publish(vx, vy, wz):
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = vx, vy, wz
        pub.publish(message)

    try:
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError('A6850 hardware node /arm service unavailable')
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if (status.get('mode') == 'ready' and status.get('walk_start_stable')
                    and status.get('send') and not status.get('stand_only')
                    and time.monotonic() - status_time[0] < .2
                    and pub.get_subscription_count() > 0):
                break
        else:
            raise RuntimeError(
                'Not ready: require fresh ready, stable and send=true, stand_only=false')
        set_arm(True)
        publish(0.0, 0.0, 0.0)
        for name, vx, vy, wz in sequence:
            print(f'切换：{name}，vx={vx}, vy={vy}, wz={wz}', flush=True)
            until = time.monotonic() + opts.segment_sec
            last_note = None
            while time.monotonic() < until:
                mode = status.get('mode')
                note = None
                if time.monotonic() - status_time[0] > 2.0:
                    note = '状态超时，继续发布以保持使能。'
                elif mode in ('fault', 'soft_hold'):
                    note = (
                        f'控制器 {mode}，继续发布、保持使能：'
                        f'{status.get("reason") or status.get("trial_reason")}'
                    )
                if note and note != last_note:
                    print(note, flush=True)
                    last_note = note
                publish(vx, vy, wz)
                rclpy.spin_once(node, timeout_sec=.02)
    except KeyboardInterrupt:
        print('演示被中断，回站。', flush=True)
    finally:
        if rclpy.ok():
            try:
                try:
                    if client.service_is_ready():
                        set_arm(False)
                finally:
                    for _ in range(3):
                        publish(0.0, 0.0, 0.0)
                        rclpy.spin_once(node, timeout_sec=.05)
            finally:
                node.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    main()
