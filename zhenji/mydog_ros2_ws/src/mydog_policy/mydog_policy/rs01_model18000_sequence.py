"""Operator-started B18000 15-second moves with marching transitions."""
import argparse
import json
import time

MOVES = (
    ('前进', .20, 0., 0.), ('后退', -.20, 0., 0.),
    ('左移', 0., .12, 0.), ('右移', 0., -.12, 0.),
    ('左转', 0., 0., .25), ('右转', 0., 0., -.25),
    ('组合', .20, .10, .25), ('反向组合', -.20, -.10, -.25),
)
SEQUENCE_LEASE_SEC = 180.
ACTION_KEYS = ('forward', 'backward', 'left', 'right', 'turn-left', 'turn-right', 'combo', 'reverse-combo')


def build_plan(action=None):
    if action == 'march':
        return [('踏步', 15., (0., 0., 0.))]
    if action is not None:
        move = MOVES[ACTION_KEYS.index(action)]
        return [(move[0], 15., tuple(move[1:]))]
    plan = [('踏步', 5., (0., 0., 0.))]
    for name, vx, vy, wz in MOVES:
        plan.extend([(name, 15., (vx, vy, wz)), ('踏步', 5., (0., 0., 0.))])
    return plan


def check_active(status, age, starting=False):
    if not 0 <= age <= .25:
        raise RuntimeError('Sequence aborted: status older than 250ms')
    if (status.get('model') != 'B18000' or not status.get('send')
            or status.get('stand_only') or not status.get('imu_calibrated')
            or not status.get('trial_armed') or status.get('walk_inhibit_latched')
            or not status.get('timing_ready')
            or status.get('mode') not in (('ready', 'walk') if starting else ('walk',))):
        raise RuntimeError('Sequence aborted: ' + str(status.get('walk_inhibit_reason')
                           or status.get('trial_reason') or status.get('mode')))


def play_plan(plan, snapshot, publish, pump, clock=time.monotonic):
    # All segment durations count from confirmed WALK, not from arm request.
    deadline = clock() + 5.
    while True:
        status, stamp = snapshot()
        check_active(status, clock()-stamp, starting=True)
        if status['mode'] == 'walk':
            break
        if clock() >= deadline:
            raise RuntimeError('Walk entry timed out')
        publish((0., 0., 0.))
        pump(.02)
    for name, seconds, vector in plan:
        print(f'{name}: {seconds:g}s vx/vy/wz={vector}', flush=True)
        deadline = clock() + seconds
        while clock() < deadline:
            status, stamp = snapshot()
            check_active(status, clock()-stamp)
            publish(vector)
            pump(.02)


def wait_for_stand(snapshot, pump, clock=time.monotonic):
    deadline = clock()+20.
    stable_since = None
    while clock() < deadline:
        pump(.02)
        status, stamp = snapshot()
        if not 0 <= clock()-stamp <= .25:
            raise RuntimeError('Return to stand unconfirmed: stale status')
        if status.get('mode') == 'fault':
            raise RuntimeError('Return to stand unconfirmed: controller fault')
        ready = (status.get('model') == 'B18000' and status.get('mode') == 'ready'
                 and status.get('send') and not status.get('trial_armed')
                 and not status.get('walk_inhibit_latched') and status.get('walk_start_stable'))
        if ready:
            if stable_since is None: stable_since = clock()
            if clock()-stable_since >= 2.:
                print('已解除行走授权，ready 且机身稳定持续 2 秒；电机仍使能。', flush=True)
                return
        else: stable_since = None
    raise RuntimeError('Return to stand unconfirmed after 20s; inspect robot/status before next command')


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-plan', action='store_true', help='Print only; no ROS/device access')
    parser.add_argument('--action', choices=('march',)+ACTION_KEYS,
                        help='One 15s action, then disarm and wait for stable ready')
    opts = parser.parse_args(args)
    plan = build_plan(opts.action)
    if opts.print_plan:
        print(json.dumps(dict(segments=plan, total_seconds=sum(s[1] for s in plan)), ensure_ascii=False, indent=2))
        return
    import rclpy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    rclpy.init()
    node = rclpy.create_node('b18000_sequence_15s')
    ns = '/mydog/model18000'
    pub = node.create_publisher(Twist, ns+'/cmd_vel', 1)
    client = node.create_client(SetBool, ns+'/arm')
    latest = [{}, float('-inf')]
    attempted = False
    def receive(msg):
        try:
            value = json.loads(msg.data)
            if isinstance(value, dict): latest[:] = [value, time.monotonic()]
        except (ValueError, TypeError): latest[:] = [{}, float('-inf')]
    node.create_subscription(String, ns+'/status', receive, 1)
    def arm(value):
        req = SetBool.Request(); req.data = value
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
        if not future.done() or future.result() is None:
            raise RuntimeError('Arm service timeout')
        if not future.result().success: raise RuntimeError(future.result().message)
    def publish(vector):
        msg = Twist(); msg.linear.x, msg.linear.y, msg.angular.z = vector
        pub.publish(msg)
    def pump(seconds):
        end = time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<end:
            rclpy.spin_once(node, timeout_sec=max(0., end-time.monotonic()))
        if not rclpy.ok(): raise RuntimeError('ROS shutdown')
    try:
        if not client.wait_for_service(timeout_sec=5.): raise RuntimeError('B18000 /arm unavailable')
        end = time.monotonic()+5.
        while time.monotonic()<end:
            pump(.05)
            s, stamp = latest
            if (time.monotonic()-stamp < .2 and s.get('mode')=='ready'
                    and s.get('model')=='B18000' and s.get('imu_calibrated')
                    and s.get('walk_start_stable') and s.get('timing_ready')
                    and s.get('send') and not s.get('stand_only')
                    and not s.get('trial_armed') and not s.get('walk_inhibit_latched')
                    and s.get('trial_lease_sec', 0) >= SEQUENCE_LEASE_SEC
                    and pub.get_subscription_count()>0): break
        else: raise RuntimeError('Require fresh calibrated ready state, timing_ready and 180s-capable B18000 node')
        attempted = True
        arm(True)
        # Wait for the post-arm status without publishing any movement vector.
        end = time.monotonic()+.2
        while not latest[0].get('trial_armed') and time.monotonic()<end: pump(.01)
        play_plan(plan, lambda: tuple(latest), publish, pump)
    except KeyboardInterrupt:
        print('Interrupted; requesting disarm', flush=True)
    finally:
        try:
            if attempted and rclpy.ok():
                arm(False)
                if opts.action:
                    wait_for_stand(lambda: tuple(latest), pump)
        finally:
            node.destroy_node()
            if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__': main()
