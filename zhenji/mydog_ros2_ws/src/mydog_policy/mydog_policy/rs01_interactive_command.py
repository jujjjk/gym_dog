"""One operator session: stand, confirm fresh ready, then start each new action."""
import json
import select
import sys
import time
from .motion_session import CommandOwner, CommandRamp, actions, ready


def main(speed_caps):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    namespace='/mydog/model23500'
    if not sys.stdin.isatty():
        raise RuntimeError('--interactive requires a terminal; use fixed commands for scripts')
    owner=CommandOwner(namespace)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=rclpy.create_node('model23500_interactive_command')
    publisher=node.create_publisher(Twist,namespace+'/cmd_vel',1)
    client=node.create_client(SetBool,namespace+'/arm')
    state={};last_status=0.;owns_arm=False;pending=None;waiting_since=None;ready_after=float("-inf")
    ramp=CommandRamp(speed_caps);moves=actions(speed_caps)
    def receive(message):
        nonlocal last_status
        try:
            value=json.loads(message.data)
            if not isinstance(value,dict):return
            state.clear();state.update(value);last_status=time.monotonic()
        except ValueError:pass
    sub=node.create_subscription(String,namespace+'/status',receive,1)
    def arm(value):
        nonlocal owns_arm
        request=SetBool.Request();request.data=value
        future=client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=2.)
        if not future.done() or future.result() is None:
            if value:owns_arm=True  # uncertain acceptance: try disarm on exit
            raise RuntimeError('Arm service timed out')
        result=future.result()
        if not result.success:raise RuntimeError(result.message)
        owns_arm=value
        print(result.message,flush=True)
    try:
        if not client.wait_for_service(timeout_sec=5.):raise RuntimeError('Controller /arm unavailable')
        print('Type a key then Enter: w forward, s backward, a left, d right, m march, '
              'q/e turn left/right, z stand, x exit. Diagonals: forward_left, forward_right, '
              'backward_left, backward_right; combined, combined_reverse. Ctrl+C returns to stand. '
              'Each new action returns to stand and waits for fresh ready first. '
              'Calibration is retained between actions. Waiting for your first action.',flush=True)
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.02)
            now=time.monotonic()
            if owns_arm and (now-last_status>.25 or state.get('mode') in ('fault','soft_hold')):
                reason=state.get('reason') or state.get('walk_inhibit_reason') or state.get('trial_reason') or 'status stale'
                arm(False);pending=None;waiting_since=None;ramp.reset()
                ready_after=time.monotonic()
                print('Motion stopped: '+str(reason)+'. Enter a new action after recovery.',flush=True)
                continue
            if select.select([sys.stdin],[],[],0)[0]:
                line=sys.stdin.readline()
                if not line:break
                key=line.strip().lower()
                if key in ('x','exit'):break
                if key in ('z','stand','stop'):
                    if owns_arm:arm(False)
                    pending=None;waiting_since=None;ramp.reset();ready_after=time.monotonic()
                    print('Returning to stand; another action will wait for ready.',flush=True)
                elif key in moves:
                    target=moves[key]
                    if owns_arm and pending == target:
                        print('Already running '+key,flush=True)
                    else:
                        if owns_arm:
                            arm(False)
                            ready_after=time.monotonic()
                        pending=target;waiting_since=time.monotonic();ramp.reset()
                        print('Requested '+key+' '+str(pending)+'; waiting for stand/ready.',flush=True)
                elif key:print('Unknown action; use w/s/a/d/m/q/e/z/x or a listed diagonal.',flush=True)
            age=now-last_status
            if not owns_arm and pending is not None:
                # Receipt after disarm is insufficient: a queued pre-disarm
                # ready message must not start the next action. Both nodes
                # run on this Jetson and share its monotonic clock.
                stamp=float(state.get('time_monotonic_s',float('-inf')))
                if (ready(state,age) and stamp>ready_after
                        and 0 <= now-stamp < .2 and publisher.get_subscription_count()>0):
                    if not state.get('continuous_commands'):raise RuntimeError('A requires continuous_commands=true')
                    arm(True);ramp.reset();waiting_since=None
                elif waiting_since is not None and now-waiting_since>30.:
                    pending=None;waiting_since=None
                    print('Still not ready: '+str({k:state.get(k) for k in
                          ('mode','walk_start_stable','stand_worst_joint','stand_max_error_rad','stand_required_error_rad',
                           'imu_calibrated','calibration_requested','timing_ready','observation_temporal_ok','reason','walk_inhibit_reason')})+
                          '. No arm request was sent; enter an action again after recovery.',flush=True)
            if owns_arm and pending is not None:
                if state.get('mode')!='walk':
                    # Do not consume the transition ramp while the controller
                    # is still waiting for its supported-start gate.
                    ramp.reset()
                vector=ramp.update(pending,time.monotonic())
                message=Twist();message.linear.x=float(vector[0]);message.linear.y=float(vector[1]);message.angular.z=float(vector[2])
                publisher.publish(message)
    except KeyboardInterrupt:pass
    finally:
        try:
            if rclpy.ok() and owns_arm:arm(False)
        finally:
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
            owner.close()
