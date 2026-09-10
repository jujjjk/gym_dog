"""ROS2 shadow inference ONLY: no motor client, enable, configure or send API.

Feedback publishers must supply acquisition timestamps in the ROS clock domain.
The legacy HTTP/serial reception timestamps must not be relabeled as such.
"""
import time
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger
from rcl_interfaces.msg import ParameterDescriptor
from .rs01_model930_core import REAL_JOINT_NAMES
from .rs01_model6850_core import Model6850Contract, Rs01Model6850Core, validate_sample_times


class Model6850ShadowNode(Node):
    def __init__(self):
        super().__init__('rs01_model6850_shadow')
        self.declare_parameter('enable_send',False,ParameterDescriptor(read_only=True))
        if self.get_parameter('enable_send').value:
            raise RuntimeError('Motor sending is not implemented/authorized by this shadow node')
        self.declare_parameter('onnx_path','')
        path=self.get_parameter('onnx_path').value
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])
        self.core=Rs01Model6850Core(session,Model6850Contract.from_onnx_session(session,path))
        self.joint=None;self.imu=None;self.command=None;self.fault=None;self.initialized=False
        self.last_motor=None;self.last_imu=None;self.last_tick=None;self.last_command=None
        self.command_time=None;self.ns='/mydog/model6850'
        self.target_pub=self.create_publisher(JointState,self.ns+'/shadow_target',10)
        self.obs_pub=self.create_publisher(Float32MultiArray,self.ns+'/observation',10)
        self.status_pub=self.create_publisher(String,self.ns+'/status',10)
        self.create_subscription(JointState,self.ns+'/joint_feedback',self.on_joint,qos_profile_sensor_data)
        self.create_subscription(Imu,self.ns+'/imu_body',self.on_imu,qos_profile_sensor_data)
        self.create_subscription(Twist,self.ns+'/cmd_vel',self.on_command,10)
        self.create_service(Trigger,self.ns+'/reset_shadow',self.reset_shadow)
        self.create_timer(.02,self.tick)
        self.get_logger().warning('SHADOW ONLY: publishes candidate targets, never commands motors')

    def on_joint(self,msg):self.joint=msg
    def on_imu(self,msg):self.imu=msg
    def on_command(self,msg):
        self.command=np.array([msg.linear.x,msg.linear.y,msg.angular.z])
        self.command_time=time.monotonic()

    def reset_shadow(self,request,response):
        self.fault=None;self.initialized=False;self.last_motor=None;self.last_imu=None;self.last_tick=None
        self.command=None;self.command_time=None;self.joint=None;self.imu=None
        response.success=True;response.message='Shadow reset; fresh feedback and a new command required'
        return response

    @staticmethod
    def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

    def tick(self):
        if self.fault or self.joint is None or self.imu is None or self.command is None:return
        try:
            wall=time.monotonic();now=self.get_clock().now().nanoseconds*1e-9
            if wall-self.command_time>.5:raise ValueError('Command timeout')
            if self.last_tick is not None and not .01<=wall-self.last_tick<=.04:raise ValueError('Policy loop timing outside guarded window')
            joint=self.joint;imu=self.imu
            if len(joint.name)!=12 or set(joint.name)!=set(REAL_JOINT_NAMES):raise ValueError('Wrong/duplicate real motor joint names')
            if len(joint.position)!=12 or len(joint.velocity)!=12:raise ValueError('Incomplete joint feedback')
            if imu.header.frame_id!='base_link':raise ValueError('IMU must already be transformed to base_link')
            if imu.orientation_covariance[0]<0 or imu.angular_velocity_covariance[0]<0:raise ValueError('Missing IMU orientation/gyro')
            motor_time=np.full(12,self.stamp(joint));imu_time=self.stamp(imu)
            validate_sample_times(now,motor_time,imu_time,self.last_motor,self.last_imu)
            order=[joint.name.index(name) for name in REAL_JOINT_NAMES]
            q,dq=self.core.mapper.real_to_policy_abs(np.array(joint.position)[order],np.array(joint.velocity)[order])
            x,y,z,w=imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w
            if not np.isfinite([x,y,z,w]).all() or abs(np.linalg.norm([x,y,z,w])-1)>.01:raise ValueError('Invalid IMU quaternion')
            norm=np.linalg.norm([x,y,z,w]);x,y,z,w=np.array([x,y,z,w])/norm
            gravity=np.array([2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1])
            yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            gyro=np.array([imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z])
            if gravity[2]>-np.cos(.6):raise ValueError('Excessive tilt')
            if not self.initialized:self.core.reset(yaw,q);self.initialized=True
            result=self.core.tick(q,dq,gyro,gravity,yaw,self.command,gait=1.)
            target=JointState();target.header.stamp=self.get_clock().now().to_msg()
            target.name=list(REAL_JOINT_NAMES);target.position=result['target_real'].tolist()
            self.target_pub.publish(target)
            obs=Float32MultiArray();obs.data=result['observation'].tolist();self.obs_pub.publish(obs)
            self.last_motor=motor_time;self.last_imu=imu_time;self.last_tick=wall
        except Exception as exc:
            self.fault=str(exc);status=String();status.data='SHADOW_FAULT: '+self.fault
            self.status_pub.publish(status);self.get_logger().error(status.data)


def main(args=None):
    rclpy.init(args=args);node=None
    try:
        node=Model6850ShadowNode();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()
