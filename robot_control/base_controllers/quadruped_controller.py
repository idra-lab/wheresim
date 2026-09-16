import matplotlib.pyplot as plt
import pinocchio as pin
from base_controllers.utils.pidManager import PidManager
from base_controllers.base_controller import BaseController
from base_controllers.utils.math_tools import *
from base_controllers.utils.joyManager import JoyManager
from base_controllers.components.whole_body_controller import WholeBodyController
from base_controllers.utils.common_functions import *
from base_controllers.components.inverse_kinematics.inv_kinematics_quadruped import InverseKinematics as AnalyticInverseKinematics
from base_controllers.components.inverse_kinematics.inv_kinematics_pinocchio import robotKinematics as PinocchioInverseKinematics
from base_controllers.components.leg_odometry.leg_odometry import LegOdometry
from base_controllers.components.rl_velocity_controller.rl_controller import RlVelocityController
from base_controllers.components.rl_velocity_controller.LocomotionPolicyWrapper import LocomotionPolicyWrapper
from termcolor import colored
import base_controllers.params as conf
from scipy.io import savemat
from datetime import datetime, timezone
import time
import traceback
#gazebo messages
from gazebo_ros import gazebo_interface
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Twist
from sensor_msgs.msg import Imu
from ros_impedance_controller.msg import EffortPid
from base_controllers.components.imu_utils import IMU_utils
from base_controllers.components.quadruped_tasks import QuadrupedTasks
from base_controllers.components.state_machine import StateMachine
from base_controllers.utils.rosbag_recorder import RosbagControlledRecorder
from optimization.srb_footstep_ocp import  SrbFootstepOcp
from optimization.lipm_to_whole_body import compute_foot_traj, interpolate_lipm_traj
from optimization.tsid_quadruped import TsidQuadruped
import optimization.aliengo_conf as com_optim_conf

class QuadrupedController(BaseController):
    def __init__(self, robot_name="hyq", launch_file=None):
        super(QuadrupedController, self).__init__(robot_name, launch_file)
        self.qj_0 = conf.robot_params[self.robot_name]['q_0']


        self.ee_frames = conf.robot_params[self.robot_name]['ee_frames']
        self.leg_names = [foot[:2] for foot in self.ee_frames]


        if not self.real_robot:
            self.gravity_comp_duration = 0.5 #1.5
            self.standup_period = 1. #3
            self.dt = conf.robot_params[self.robot_name]['dt']
        else:
            self.gravity_comp_duration = 1.5
            self.standup_period = 3.
            print(colored("overriding default dt: quadruped controller for real robot runs at 250Hz!","red"))
            self.dt = 0.004


    #####################
    # OVERRIDEN METHODS #
    #####################
    # initVars
    # logData
    # startupProcedure

    def initSubscribers(self):
        self.sub_jstate = ros.Subscriber("/" + self.robot_name + "/joint_states", JointState, callback=self._receive_jstate, queue_size=1, tcp_nodelay=True)
        self.sub_pid_effort = ros.Subscriber("/" + self.robot_name + "/effort_pid", EffortPid, callback=self._receive_pid_effort, queue_size=1, tcp_nodelay=True)
        self.sub_imu = ros.Subscriber("/" + self.robot_name + "/trunk_imu", Imu,  callback=self._receive_imu, queue_size=1, tcp_nodelay=True)

        if self.state_estimation == 'ekf':  # use pronto for state estimation
            #  stateest node requires publication of msg type sensor:IMU in topic aliengo/imu we remap
            startNode(package="topic_tools", executable="relay", args="/" + self.robot_name + "/trunk_imu" + "  " + "/" + self.robot_name + "/imu", name="trunk_imu_to_imu")
            self.pronto_config = "aliengo_state_estimator_sim.yaml"
            from pronto_msgs.msg import QuadrupedStance, QuadrupedForceTorqueSensors
            self.pronto_contacts_sub = ros.Subscriber("/state_estimator_pronto/stance", QuadrupedStance, callback=self._receive_pronto_contacts, queue_size=1, tcp_nodelay=True)
            self.sub_pose = ros.Subscriber("/state_estimator_pronto/odom", Odometry, callback=self._receive_pose, queue_size=1, tcp_nodelay=True)
            #these on real robot are published by hw interface
            self.pub_feet_forces = ros.Publisher("/" + self.robot_name + "/feet_forces", QuadrupedForceTorqueSensors, queue_size=1, tcp_nodelay=True)

        elif self.state_estimation=='ground_truth':
            self.sub_pose = ros.Subscriber("/" + self.robot_name + "/ground_truth", Odometry,  callback=self._receive_pose,  queue_size=1, tcp_nodelay=True)
        else:
            print(f"state_estimation type not known {self.state_estimation}")

        if self.use_ground_truth_contacts:
            self.sub_contact_lf = ros.Subscriber("/" + self.robot_name + "/lf_foot_bumper", ContactsState,
                                                 callback=self._receive_contact_lf, queue_size=1, buff_size=2 ** 24,
                                                 tcp_nodelay=True)
            self.sub_contact_rf = ros.Subscriber("/" + self.robot_name + "/rf_foot_bumper", ContactsState,
                                                 callback=self._receive_contact_rf, queue_size=1, buff_size=2 ** 24,
                                                 tcp_nodelay=True)
            self.sub_contact_lh = ros.Subscriber("/" + self.robot_name + "/lh_foot_bumper", ContactsState,
                                                 callback=self._receive_contact_lh, queue_size=1, buff_size=2 ** 24,
                                                 tcp_nodelay=True)
            self.sub_contact_rh = ros.Subscriber("/" + self.robot_name + "/rh_foot_bumper", ContactsState,
                                                 callback=self._receive_contact_rh, queue_size=1, buff_size=2 ** 24,
                                                 tcp_nodelay=True)

    def _receive_imu(self, msg):
        self.quaternion[0] = msg.orientation.x
        self.quaternion[1] = msg.orientation.y
        self.quaternion[2] = msg.orientation.z
        self.quaternion[3] = msg.orientation.w

        self.euler = np.array(euler_from_quaternion(self.quaternion))
        #euler angles
        self.basePoseW[self.u.sp_crd["AX"]] = self.euler[0]
        self.basePoseW[self.u.sp_crd["AY"]] = self.euler[1]
        self.basePoseW[self.u.sp_crd["AZ"]] = self.euler[2]

        # compute orientation matrix
        self.b_R_w = self.math_utils.rpyToRot(self.euler)
        self.angVelB[0] = msg.angular_velocity.x
        self.angVelB[1] = msg.angular_velocity.y
        self.angVelB[2] = msg.angular_velocity.z
        # angular part of twist
        self.baseTwistW[3:] = self.b_R_w.T.dot(self.angVelB)

        # linear acceleration
        self.baseLinAccB[0] = msg.linear_acceleration.x
        self.baseLinAccB[1] = msg.linear_acceleration.y
        self.baseLinAccB[2] = msg.linear_acceleration.z

        # baseLinAccW is without gravity
        self.baseLinAccW = self.b_R_w.T @ (self.baseLinAccB - self.imu_utils.IMU_accelerometer_bias) - self.imu_utils.g0

        # get estimates of base position and linear twist by odometry
        if self.state_estimation == 'odometry':
            self.basePoseW[self.u.sp_crd["LX"]] = self.basePoseW_legOdom[0]
            self.basePoseW[self.u.sp_crd["LY"]] = self.basePoseW_legOdom[1]
            self.basePoseW[self.u.sp_crd["LZ"]] = self.basePoseW_legOdom[2]

            self.baseTwistW[self.u.sp_crd["LX"]] = self.baseTwistW_legOdom[0]
            self.baseTwistW[self.u.sp_crd["LY"]] = self.baseTwistW_legOdom[1]
            self.baseTwistW[self.u.sp_crd["LZ"]] = self.baseTwistW_legOdom[2]

    def initVars(self):
        super().initVars()
        self.q_des = np.zeros_like(self.q)

        # mocap filter coeff
        ta = 0.15
        self.beta = self.dt / (self.dt + ta)
        self.basePoseW_f_old = np.zeros(6)
        self.basePoseW_f = np.zeros(6)

        self.imu_utils = IMU_utils(dt=conf.robot_params[self.robot_name]['dt'])
        #pinocchio based
        self.ikin = PinocchioInverseKinematics(self.robot, conf.robot_params[self.robot_name]['ee_frames'])
        self.IK = AnalyticInverseKinematics(self.robot)
        self.leg_odom = LegOdometry(self.robot, self.real_robot)
        self.legConfig = {}
        if 'solo' in self.robot_name or  self.robot_name == 'hyq' or self.robot_name == 'anymal_d':  # either solo or solo_fw
            self.legConfig['lf'] = ['HipDown', 'KneeInward']
            self.legConfig['lh'] = ['HipDown', 'KneeInward']
            self.legConfig['rf'] = ['HipDown', 'KneeInward']
            self.legConfig['rh'] = ['HipDown', 'KneeInward']

        elif self.robot_name == 'aliengo' or self.robot_name == 'go1' or self.robot_name == 'go2':
            self.legConfig['lf'] = ['HipDown', 'KneeInward']
            self.legConfig['lh'] = ['HipDown', 'KneeOutward']
            self.legConfig['rf'] = ['HipDown', 'KneeInward']
            self.legConfig['rh'] = ['HipDown', 'KneeOutward']

        else:
            assert False, 'leg configuration is not defined for ' + self.robot_name

        self.euler = np.zeros(3)
        # some extra variables

        self.tau_fb = np.zeros(self.robot.na)
        self.tau_ffwd = np.zeros(self.robot.na)
        self.tau_des = np.zeros(self.robot.na)

        self.basePoseW_des = np.zeros(6) * np.nan
        self.baseTwistW_des = np.zeros(6) * np.nan
        self.baseAccW_des = np.zeros(6) * np.nan

        self.comPoseW_des = np.zeros(6) * np.nan
        self.comTwistW_des = np.zeros(6) * np.nan
        self.comAccW_des = np.zeros(6) * np.nan

        self.comPosB = np.zeros(3) * np.nan
        self.comVelB = np.zeros(3) * np.nan

        self.basePoseW_legOdom = np.zeros(3) #* np.nan
        self.baseTwistW_legOdom = np.zeros(3) #* np.nan
        ###W
        self.g_mag = np.linalg.norm(self.robot.model.gravity.vector)

        self.grForcesW_des = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesW_wbc = np.empty(3 * self.robot.nee) * np.nan
        self.grForcesB = np.empty(3 * self.robot.nee) * np.nan #not used
        self.grForcesB_ffwd = np.empty(3 * self.robot.nee) * np.nan

        # load gains
        if self.real_robot:
            real_str = '_real'
        else:
            real_str = ''

        # stand alone joint pid
        self.kp_j = conf.robot_params[self.robot_name].get('kp'+real_str, np.zeros(self.robot.na))
        self.kd_j = conf.robot_params[self.robot_name].get('kd'+real_str, np.zeros(self.robot.na))
        self.ki_j = conf.robot_params[self.robot_name].get('ki'+real_str, np.zeros(self.robot.na))

        ###W
        # virtual impedance wrench control
        self.kp_lin = np.diag(conf.robot_params[self.robot_name].get('kp_lin'+real_str, np.zeros(3)))
        self.kd_lin = np.diag(conf.robot_params[self.robot_name].get('kd_lin'+real_str, np.zeros(3)))

        self.kp_ang = np.diag(conf.robot_params[self.robot_name].get('kp_ang'+real_str, np.zeros(3)))
        self.kd_ang = np.diag(conf.robot_params[self.robot_name].get('kd_ang'+real_str, np.zeros(3)))

        # updated in WBC
        self.kp_linW = np.zeros_like(self.kp_lin)
        self.kd_linW = np.zeros_like(self.kd_lin)

        self.kp_angW = np.zeros_like(self.kp_ang)
        self.kd_angW = np.zeros_like(self.kd_ang)

        # joint pid with wbc
        self.kp_wbc_j = conf.robot_params[self.robot_name].get('kp_wbc'+real_str, np.zeros(self.robot.na))
        self.kd_wbc_j = conf.robot_params[self.robot_name].get('kd_wbc'+real_str, np.zeros(self.robot.na))
        self.ki_wbc_j = conf.robot_params[self.robot_name].get('ki_wbc'+real_str, np.zeros(self.robot.na))


        self.wrench_fbW  = np.zeros(6)
        self.wrench_ffW  = np.zeros(6)
        self.wrench_gW   = np.zeros(6)
        self.wrench_gW[self.u.sp_crd["LZ"]] = self.robot.robotMass * self.g_mag
        self.wrench_desW = np.zeros(6)

        self.wrench_fbW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_ffW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_gW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)
        self.wrench_desW_log = np.full( (6, conf.robot_params[self.robot_name]['buffer_size'] ), np.nan)

        self.NEMatrix = np.zeros([6, 3*self.robot.nee]) # Newton-Euler matrix

        ###W
        self.wbc = WholeBodyController(conf.robot_params[self.robot_name], self.real_robot, self.robot)

        self.force_th = conf.robot_params[self.robot_name].get('force_th', 0.)
        self.contact_th = conf.robot_params[self.robot_name].get('contact_th', 0.)

        self.W_vel_contacts_des = self.u.full_listOfArrays(4, 3)
        self.B_vel_contacts_des = self.u.full_listOfArrays(4, 3)

        # imu
        self.baseLinAccB = np.full(3, np.nan)
        self.baseLinAccW = np.full(3, np.nan)


        self.comPosB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comVelB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comPoseW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comTwistW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comPoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.comTwistW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.cop_des = np.full(2, np.nan)
        self.cop_act = np.full(2, np.nan)
        self.cop_des_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.cop_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.comVelW_leg_odom = np.full((3), np.nan)
        self.comVelW_leg_odom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)


        self.basePoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseTwistW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.basePoseW_legOdom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseTwistW_legOdom_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.tau_fb_log = np.full((self.robot.na, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.tau_des_log = np.full((self.robot.na, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)


        self.grForcesB_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.grForcesW_gt_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.grForcesW_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.grForcesW_wbc_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.W_contacts_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.W_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.B_contacts_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.B_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.B_vel_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.W_vel_contacts_des_log = np.full((3 * self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.contact_state_log = np.full((self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.stance_legs_log = np.full((self.robot.nee, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.baseLinAccW_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.baseLinAccB_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.baseLinTwistImuW_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.angVelB = np.zeros(3)

        # robot height is the height of the robot base frame in home configuration
        self.robot_height = 0.

        neutral_fb_jointstate = np.hstack([pin.neutral(self.robot.model)[0:7], self.u.mapToRos(conf.robot_params[self.robot_name]['q_0']) ])

        self.robot.forwardKinematics(neutral_fb_jointstate)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        for id in self.robot.getEndEffectorsFrameId:
            self.robot_height += self.robot.data.oMf[id].translation[2]
        self.robot_height /= -4.
        print(colored(f"Robot height correspontent to q0 configuration is {self.robot_height+0.02}","red")) #Includes the foot radius
        self.loop_time_log = np.full((conf.robot_params[self.robot_name]['buffer_size']), np.nan)

        #safety layer
        self.gracefulCollapseFlag = False
        self.alphaCollapse = 1.0
        self.collapseTime = 0.7

        self.pronto_contacts = np.zeros(4)

        #friction coefficient
        self.mu = 0.8

    def logData(self):
        # full with new values
        self.comPosB_log[:, self.log_counter] = self.comPosB
        self.comVelB_log[:, self.log_counter] = self.comVelB
        self.comPoseW_log[:, self.log_counter] = self.comPoseW
        self.comTwistW_log[:, self.log_counter] = self.comTwistW
        self.comPoseW_des_log[:, self.log_counter] = self.comPoseW_des
        self.comTwistW_des_log[:, self.log_counter] = self.comTwistW_des
        self.cop_des_log[:, self.log_counter] = self.cop_des
        self.cop_log[:, self.log_counter] = self.cop_act
        self.basePoseW_log[:, self.log_counter] = self.basePoseW
        self.baseTwistW_log[:, self.log_counter] = self.baseTwistW
        self.basePoseW_des_log[:, self.log_counter] = self.basePoseW_des
        self.baseTwistW_des_log[:, self.log_counter] = self.baseTwistW_des
        self.basePoseW_legOdom_log[:, self.log_counter] = self.basePoseW_legOdom
        self.baseTwistW_legOdom_log[:, self.log_counter] = self.baseTwistW_legOdom
        self.q_des_log[:, self.log_counter] = self.q_des
        self.q_log[:, self.log_counter] = self.q
        self.qd_des_log[:, self.log_counter] = self.qd_des
        self.qd_log[:, self.log_counter] = self.qd
        self.tau_fb_log[:, self.log_counter] = self.tau_fb
        self.tau_ffwd_log[:, self.log_counter] = self.tau_ffwd

        self.tau_des = self.tau_ffwd + self.tau_fb
        self.tau_des_log[:, self.log_counter] = self.tau_des
        self.tau_log[:, self.log_counter] = self.tau
        self.grForcesW_log[:, self.log_counter] = self.grForcesW
        self.grForcesW_des_log[:, self.log_counter] = self.grForcesW_des
        self.grForcesW_wbc_log[:, self.log_counter] = self.grForcesW_wbc
        self.grForcesW_gt_log[:, self.log_counter] = self.grForcesW_gt
        self.grForcesB_log[:, self.log_counter] = self.grForcesB
        self.contact_state_log[:, self.log_counter] = self.contact_state
        self.stance_legs_log[:, self.log_counter] = self.stance_legs

        self.baseLinAccW_log[:, self.log_counter] = self.baseLinAccW
        self.baseLinAccB_log[:, self.log_counter] = self.baseLinAccB

        self.comVelW_leg_odom_log[:, self.log_counter] = self.comVelW_leg_odom

        for leg in range(4):
            start = 3 * leg
            end = 3 * (leg+1)
            self.B_contacts_log[start:end, self.log_counter] = self.B_contacts[leg]
            self.B_contacts_des_log[start:end, self.log_counter] = self.B_contacts_des[leg]

            self.W_contacts_log[start:end, self.log_counter] = self.W_contacts[leg]
            self.W_contacts_des_log[start:end, self.log_counter] = self.W_contacts_des[leg]

            self.B_vel_contacts_des_log[start:end, self.log_counter] = self.B_vel_contacts_des[leg]
            self.W_vel_contacts_des_log[start:end, self.log_counter] = self.W_vel_contacts_des[leg]

        self.baseLinTwistImuW_log[:, self.log_counter] = self.imu_utils.baseLinTwistImuW

        ###W
        self.wrench_fbW_log[:, self.log_counter] = self.wrench_fbW
        self.wrench_ffW_log[:, self.log_counter] = self.wrench_ffW
        self.wrench_gW_log[:, self.log_counter] = self.wrench_gW
        self.wrench_desW_log[:, self.log_counter] = self.wrench_desW
        self.time_log[self.log_counter] = self.time
        self.loop_time_log[self.log_counter] = self.loop_time

        self.log_counter += 1
        self.log_counter %= conf.robot_params[self.robot_name]['buffer_size']
    
    def check_faulty_ping(self, ip="192.168.1.1"):
        response = os.system("ping -c 1 " + ip)
        # and then check the response...
        if response == 0:
            pingstatus = f"Robot Network Active: pinging {ip} successful"
        else:
            pingstatus = f"Robot Network not Active, cannot ping {ip}, create a local network with gateway {ip}"
        print(colored(pingstatus, "red"))
        return response

    def startController(self, world_name=None, xacro_path=None,   use_ground_truth_contacts=True, additional_args=[]):

        if self.real_robot == False:
            self.use_ground_truth_contacts = use_ground_truth_contacts
        else:
            self.use_ground_truth_contacts = False
            if self.check_faulty_ping(conf.robot_params[self.robot_name]['ip']):
                sys.exit()

        self.start()                               # as a thread

        self.go0_conf = 'home'
        if additional_args is not None:
            for arg in additional_args:
                if 'go0_conf:=' in arg:
                    self.go0_conf = arg.replace('go0_conf:=', '')


        additional_args.append("load_force_sensors:="+str(not self.use_ground_truth_contacts).lower())
        if self.use_ground_truth_contacts and (world_name is None or not 'slow' in world_name):
            print('Cannot use ground truth contact with not slow world file')
            print('Set world file to slow.world')
            world_name = 'slow.world'

        self.startSimulator(world_name=world_name, additional_args=additional_args)            # run gazebo
        if world_name is None:
            self.world_name_str = ''
        else:
            self.world_name_str = world_name
        if 'camera' in self.world_name_str:
            # check if some old jpg are still in /tmp
            # this command prevent for Argument list too long in bash http://mywiki.wooledge.org/BashFAQ/095
            print(colored('Removing jpg files', 'blue'), flush=True)
            remove_jpg_cmd = 'for f in /tmp/camera_save/*; do rm "$f"; done'
            os.system(remove_jpg_cmd)
            print(colored('Jpg files removed', 'blue'), flush=True)
        self.loadModelAndPublishers(xacro_path)    # load robot and all the publishers
        #self.resetGravity(True)
        self.initVars()                            # overloaded method
        self.initSubscribers()
        self.rate = ros.Rate(1 / self.dt)
        print(colored("Started QuadrupedController", "blue"))

    def resetRobot(self, basePoseDes=np.array([0, 0, 0.3, 0., 0., 0.]), baseTwistDes=None,
                   freeze_base=False, settle_timeout=2.0):
        """
        Teleport the robot to basePoseDes with joints at q_0 and leave it ready to be controlled.
        Simulation-only replacement for startupProcedure(): instead of the gradual gravity-compensated
        stand-up sequence, it places the robot directly in its nominal configuration. Meant to be used
        for fast resets (e.g. between RL episodes) rather than for the real robot power-on sequence.
        """
        assert not self.real_robot, "resetRobot() teleports the robot through Gazebo services, it is simulation-only"

        if baseTwistDes is None:
            baseTwistDes = np.zeros(6)

        # target joint state the PD controller should hold once the robot is teleported
        self.q_des = self.qj_0.copy()
        self.qd_des = np.zeros(self.robot.na)
        self.tau_ffwd = np.zeros(self.robot.na)

        # make sure the PID exists and is back at nominal gains (e.g. after gracefulCollapse ramped them down)
        if getattr(self, 'pid', None) is None:
            self.pid = PidManager(self.joint_names)
        self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)

        # freeze gravity before teleporting so the robot cannot start falling before joints/base are in place
        self.freezeBase(False, basePoseW=basePoseDes, baseTwistW=baseTwistDes)
        gazebo_interface.set_model_configuration_client(self.robot_name, '', self.joint_names, self.qj_0, '/gazebo')
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        # re-anchor the leg odometry to the teleported feet, otherwise it keeps using the pre-reset ones
        self.leg_odom.reset(np.hstack([self.u.linPart(basePoseDes), self.quaternion, self.q]))
        self.imu_utils.baseLinTwistImuW = self.u.linPart(baseTwistDes).copy()
        # release (or keep frozen, for debug/tasks that need it) gravity now that the robot is settled
        self.freezeBase(False, basePoseW=basePoseDes, baseTwistW=baseTwistDes)
        self.updateKinematics()

    def Hframe2World(self, poseH, dposeH=None, ddposeH=None):
        # returns variables from Hframe to World frame
        # dposeH is the rate (d/dt poseH) and ddposeH its time derivative
        # dposeH is mapped into twist and ddposeH into acceleration
        w_R_des_hf = pin.rpy.rpyToMatrix(0, 0, self.u.angPart(self.basePoseW)[2])

        poseW = np.empty(6)
        poseW[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = w_R_des_hf @ self.u.linPart(poseH)
        poseW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = self.u.angPart(poseH)
        #poseW[self.u.sp_crd['AZ']] += np.pi/2#self.u.angPart(self.basePoseW)[2]

        if dposeH is not None:
            twistW = np.empty(6)

            twistW[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] =  w_R_des_hf @ self.u.linPart(dposeH)

            # map euler rates into omega
            Jomega = self.math_utils.Tomega(self.u.angPart(self.basePoseW))
            twistW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega @ (self.u.angPart(dposeH))

            if ddposeH is not None:
                accW = np.empty(6)
                accW[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = w_R_des_hf @ self.u.linPart(ddposeH)
                accW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = self.u.angPart(ddposeH)

                # compute w_omega_dot =  Jomega* euler_rates_dot + Jomega_dot*euler_rates (Jomega already computed, see above)
                Jomega_dot = self.math_utils.Tomega_dot(self.u.angPart(self.basePoseW), self.u.angPart(self.baseTwistW))
                accW[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega @ self.u.angPart(ddposeH) + \
                                                                    Jomega_dot @ self.u.angPart(dposeH)
                return poseW, twistW, accW

            return poseW, twistW

        return poseW


    def World2Hframe(self, poseW, twistW=None, accW=None):
        # inverse of the previous function
        # returns variables from World frame to Hframe
        # twist is mapped into dposeH and acceleration into ddposeH
        hf_R_des_w = pin.rpy.rpyToMatrix(0, 0, self.u.angPart(poseW)[2]).T

        poseH = np.empty(6)
        poseH[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = hf_R_des_w @ self.u.linPart(poseW)
        poseH[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = self.u.angPart(poseW)[0:3]
        # poseH[self.u.sp_crd['AZ']] = 0.

        if twistW is not None:
            dposeH = np.empty(6)
            dposeH[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] =  hf_R_des_w @ self.u.linPart(twistW)
            # map omega into euler rate
            Jomega_inv = self.math_utils.Tomega_inv(self.u.angPart(poseW))
            dposeH[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega_inv @ self.u.angPart(twistW)

            if accW is not None:
                ddposeH = np.empty(6)
                ddposeH[self.u.sp_crd['LX']:self.u.sp_crd['LX'] + 3] = hf_R_des_w @ self.u.linPart(accW)
                # compute euler_rates_dot = Jomega_inv *( w_omega_dot - Jomega_dot*euler_rates) (Jomega_inv already computed, see above)
                Jomega_dot = self.math_utils.Tomega_dot(self.u.angPart(poseW), self.u.angPart(twistW))

                ddposeH[self.u.sp_crd['AX']:self.u.sp_crd['AX'] + 3] = Jomega_inv @ (self.u.angPart(accW) - Jomega_dot @ self.u.angPart(dposeH) )
                return poseH, dposeH, ddposeH

            return poseH, dposeH

        return poseH




    def Wcom2Wbase_des(self):
        # suppose comPose/Twist des and W_contacts_des are set
        # base ref in W
        b_R_w_des = pin.rpy.rpyToMatrix(self.u.angPart(self.comPoseW_des)).T
        omega_skew = pin.skew(self.u.angPart(self.comTwistW_des))

        self.basePoseW_des = self.comPoseW_des.copy()
        self.basePoseW_des[:3] -= b_R_w_des.T @ self.comPosB

        self.baseTwistW_des = self.comTwistW_des.copy()
        # NOTE: comPosB must be rotated to world BEFORE crossing it with the (world-frame)
        # angular velocity: omega_skew @ (R @ comPosB), not R @ (omega_skew @ comPosB) -
        # the rotation and the cross product do not commute
        self.baseTwistW_des[:3] -= omega_skew @ (b_R_w_des.T @ self.comPosB) + b_R_w_des.T @ self.comVelB

    def Wbase2Bcontact_des(self):
        b_R_w_des = pin.rpy.rpyToMatrix(self.u.angPart(self.basePoseW_des)).T
        omega_skew = pin.skew(self.u.angPart(self.baseTwistW_des))
        # feet ref in B
        for leg in range(4):
            self.B_contacts_des[leg] = b_R_w_des @ (self.W_contacts_des[leg] - self.u.linPart(self.basePoseW_des))
            self.B_vel_contacts_des[leg] = b_R_w_des @ (
                    omega_skew.T @ (self.W_contacts_des[leg] - self.u.linPart(self.basePoseW_des))
                    - self.u.linPart(self.baseTwistW_des))
            # self.B_vel_contacts_des[leg] = 5 * (self.B_contacts_des[leg]-self.B_contacts[leg])



    def Wbase2Joints_des(self):
        # before the first call, please set
        # q_des = q.copy()
        # qd_des[:] = 0.
        # suppose WbasePose/Twist des and W_contacts_des are set
        self.Wbase2Bcontact_des()

        # time = 0, 1, 2, 3, ...
        # time = 0 -> q_des = ik(),         qd_des = const.
        # time = 1 -> q_des += qd_des * dt, qd_des = const.
        # time = 2 -> q_des = const,        qd_des = diff_ik()
        # time = 3 -> q_des += qd_des * dt, qd_des = const.
        # ...

        if self.log_counter % 4 == 0:
            for leg in range(4):
                q_des_leg, isFeasible = self.IK.ik_leg(self.B_contacts_des[leg],
                                                       self.leg_names[leg],
                                                       self.legConfig[self.leg_names[leg]][0],
                                                       self.legConfig[self.leg_names[leg]][1])
                if isFeasible:
                    self.u.setLegJointState(leg, q_des_leg, self.q_des)

        elif self.log_counter % 4 == 2:
            for leg in range(4):
                qd_leg_des = self.IK.diff_ik_leg(q_des=self.q_des,
                                                 B_v_foot=self.B_vel_contacts_des[leg],
                                                 leg=self.leg_names[leg],
                                                 update=leg == 0)  # update Jacobians only with the first leg

                # qd_leg_des = self.J_inv[leg] @ self.B_vel_contacts_des[leg]

                self.u.setLegJointState(leg, qd_leg_des, self.qd_des)
        else:
            self.q_des += self.qd_des * self.dt

    def Wcom2Joints_des(self):
        # before the first call, please set
        # q_des = q.copy()
        # qd_des[:] = 0.
        # suppose WcomPose/Twist des and W_contacts_des are set
        # base ref in W
        self.Wcom2Wbase_des()
        self.Wbase2Joints_des()
        

    def support_poly(self, contacts):
        # Wcontacts: lf, rf, lh, rh
        sp = {}
        # CCW order
        sides_order = {'F': [1, 0], 'L': [0, 2], 'H': [2, 3], 'R': [3, 1]}
        for side in sides_order: # side is the key of sides_order
            p0 = contacts[sides_order[side][0]][0:2]
            p1 = contacts[sides_order[side][1]][0:2]
            m, q = self.line2points2D(p0, p1)
            sp['line'+side] = {'m': m, 'q': q, 'p0':p0, 'p1':p1}
        return sp

    @staticmethod
    def line2points2D(p0, p1):
        # line defined as y= mx+q
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        q = p0[1] - m * p0[0]
        return m, q

    def computeCoP(self):
        # actual CoP: weighted average of the stance feet XY positions, weighted by the vertical grf
        num = np.zeros(2)
        den = 0.
        for leg in range(4):
            if self.contact_state[leg]:
                fz = self.u.getLegJointState(leg, self.grForcesW)[2]
                num += fz * self.W_contacts[leg][:2]
                den += fz
        if den > self.force_th:
            return num / den
        return np.full(2, np.nan)

    def send_command(self, q_des=None, qd_des=None, tau_ffwd=None, log_data_in_send_command = False):
        # q_des, qd_des, and tau_ffwd have dimension 12
        # and are ordered as on the robot

        if q_des is not None:
            self.q_des = q_des

        if qd_des is not None:
            self.qd_des = qd_des

        if tau_ffwd is not None:
            self.tau_ffwd = tau_ffwd

        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        # log variables
        if log_data_in_send_command:
            self.logData()
        self.rate.sleep()
        self.sync_check()
        self.time = np.round(self.time + self.dt, 4)


    def visualizeContacts(self, delete_markers=False):
        for legid in self.u.leg_map.keys():

            leg = self.u.leg_map[legid]
            if self.contact_state[leg]:

                #friciton cones
                self.ros_pub.add_cone(self.W_contacts[leg], np.array([0, 0, 1.]), self.mu, height=0.15, color="blue")

                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW/ (6*self.robot.robotMass)),
                                       "green")
                
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW_des/ (6*self.robot.robotMass)),
                                       "blue")
                
                #self.ros_pub.add_marker(self.W_contacts[leg], radius=0.1)
            else:
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       np.zeros(3),
                                       "green", scale=0.0001)
                #self.ros_pub.add_marker(self.W_contacts[leg], radius=0.001)

            if (self.use_ground_truth_contacts):
                self.ros_pub.add_arrow(self.W_contacts[leg],
                                       self.u.getLegJointState(leg, self.grForcesW_gt / (6 * self.robot.robotMass)),
                                       "red")


            #
            self.ros_pub.add_marker(self.W_contacts[leg], radius=0.001)



        self.ros_pub.add_polygon([self.B_contacts[0],
                                  self.B_contacts[1],
                                  self.B_contacts[3],
                                  self.B_contacts[2],
                                  self.B_contacts[0] ], "red", visual_frame="base_link")

        self.ros_pub.publishVisual(delete_markers=delete_markers)

    def updateKinematics(self, update_legOdom=True, noise=None):
        if noise is not None:
            if 'qd' in noise:
                self.qd += noise['qd'].draw()
            if 'tau' in noise:
                self.tau += noise['tau'].draw()
        self.basePoseW_legOdom, self.baseTwistW_legOdom = self.leg_odom.base_in_world(contact_state=self.contact_state,
                                                                                      B_contacts=self.B_contacts,
                                                                                      b_R_w=self.b_R_w,
                                                                                      wJ=self.wJ,
                                                                                      ang_vel=self.b_R_w.T.dot(self.angVelB),
                                                                                      qd=self.qd,
                                                                                      update_legOdom=update_legOdom)
        self.imu_utils.compute_lin_vel(self.baseLinAccW, self.loop_time)
        super(QuadrupedController, self).updateKinematics()

        #publish contact forces in the topic for pronto
        if not self.real_robot and self.state_estimation == 'pronto':
            from pronto_msgs.msg import QuadrupedForceTorqueSensors
            msg = QuadrupedForceTorqueSensors()
            msg.lf.force.z = self.u.getLegJointState(self.u.leg_map["LF"], self.grForcesW)[2] #z component
            msg.rf.force.z = self.u.getLegJointState(self.u.leg_map["RF"], self.grForcesW)[2]  # z component
            msg.lh.force.z = self.u.getLegJointState(self.u.leg_map["LH"], self.grForcesW)[2]  # z component
            msg.rh.force.z = self.u.getLegJointState(self.u.leg_map["RH"], self.grForcesW)[2] # z component
            self.pub_feet_forces.publish(msg)


    # homning helper functions
    def IMUBiasEstimation(self, sm, time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Imu bias estimation", "blue"))
        #on loop
        #print(f"Estimating Bias {time}")
        self.updateKinematics()
        self.imu_utils.IMU_bias_estimation(self.b_R_w, self.baseLinAccB)
        self.tau_ffwd[:] = 0.
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        self.imu_utils.counter+=1
        #event driven termination
        if self.imu_utils.counter >= self.imu_utils.timeout:
            print("Estimating Bias Accomplished → next state")
            sm.next(time)

    def goFoldConfig(self, sm , time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Going to fold configuration", "blue"))
            # go from where you are to q_fold
            self.q_ref = np.zeros_like(self.q)
            # Going to fold config
            for i in range(12):
               if (i % 3) != 0:
                    self.q_ref[i] = conf.robot_params[self.robot_name]['q_fold'][i]
            self.q_init = self.q.copy()

        # on loop
        #print(f"Go fold...{time}")
        self.updateKinematics()
        self.tau_ffwd[:] = 0.
        alpha  = sm.timer.get_elapsed_time(time)/sm.get_state_duration()
        self.q_des = (1 - alpha) * self.q_init + alpha * self.q_ref
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        #termination
        if sm.timer.is_elapsed(time):
            print("Go fold Accomplished → next state")
            sm.next(time)

    # helpers for homing procedure

    def contactsAchieved(self):
        contacts_achieved = True
        for leg in range(4):
            if self.B_contacts[leg][2] > -0.04:
                contacts_achieved = False
            elif not self.contact_state[leg]:
                contacts_achieved = False
        return  contacts_achieved

    def searchingContacts(self, sm, time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "Searching contacts", "blue"))
            # sample feet position
            self.B_feet_vel = self.u.full_listOfArrays(4, 3, 0, 0.)
            neutral_fb_jointstate = np.hstack((pin.neutral(self.robot.model)[0:7], self.q))
            pin.forwardKinematics(self.robot.model, self.robot.data, neutral_fb_jointstate)
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            for leg in range(4):
                foot = conf.robot_params[self.robot_name]['ee_frames'][leg]
                foot_id = self.robot.model.getFrameId(foot)
                self.B_contacts_des[leg] = self.robot.data.oMf[foot_id].translation.copy()
            # increase of the motion (m/s)
            if self.real_robot:
                self.delta_z = 0.005
            else:
                self.delta_z = 0.1
        #on loop
        h_R_w = self.b_R_w @ pin.rpy.rpyToMatrix(0, 0, self.u.angPart(self.basePoseW)[2])
        for leg in range(4):
            # update feet task to extend feet to acquire contact
            self.B_contacts_des[leg][2] -= self.delta_z * self.dt
            q_des_leg, isFeasible = self.IK.ik_leg(h_R_w.T @ self.B_contacts_des[leg],
                                                   self.leg_names[leg],
                                                   self.legConfig[self.leg_names[leg]][0],
                                                   self.legConfig[self.leg_names[leg]][1])
            self.u.setLegJointState(leg, q_des_leg, self.q_des)
        for leg in range(4):
            self.B_feet_vel[leg][2] = -self.delta_z
            qd_leg_des = self.IK.diff_ik_leg(q_des=self.q_des,
                                             B_v_foot=self.B_feet_vel[leg],
                                             leg=self.leg_names[leg],
                                             update=leg == 0)
            self.u.setLegJointState(leg, qd_leg_des, self.qd_des)
        self.tau_ffwd[:] = 0.
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        #terminate
        if self.contactsAchieved():
            print("Searching contacts → next state")
            self.basePoseW_des = self.basePoseW.copy()
            self.baseTwistW_des[:] = 0
            self.comPoseW_des = self.comPoseW.copy()
            self.comTwistW_des[:] = 0
            self.q_des = self.q.copy()
            self.qd_des[:] = 0
            # base height
            base_height = 0.
            for leg in range(4):
                base_height -= self.B_contacts[leg][2]
            self.leg_odom.reset(np.hstack([0., 0., base_height / 4, self.quaternion, self.q]))
            sm.next(time)

    def applyGravityComp(self, sm, time):
        if sm.first_time:
            print(colored("[startupProcedure t: " + str(self.time[0]) + "s] appling gravity compensation", "blue"))

        #on loop
        #print(f"gravity comp...{time}")
        alpha = sm.timer.get_elapsed_time(time)/sm.get_state_duration()
        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,
                                                                             self.wJ,
                                                                             self.h_joints,
                                                                             self.basePoseW)
        #self.visualizeContacts()
        self.tau_ffwd *= alpha
        self.qd_des[:] = 0
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

        #termination
        if sm.timer.is_elapsed(time):
            print("gravity comp Accomplished → next state")
            sm.next(time)

    def standUp(self, sm, time):
        if sm.first_time:
            print(colored(f"[startupProcedure to make RobotHeight {self.robot_height+0.02} t: " + str(self.time[0]) + "s] moving to desired height (" + str(np.around(self.robot_height+0.02, 3)) +" m)", "blue"))
            #compute desired feet position associated to desired robot height
            #1 sample B_contacts_des = actual
            self.B_contacts_sampled = self.u.full_listOfArrays(4, 3)
            for leg in range(4):
                self.B_contacts_sampled[leg] = self.B_contacts[leg].copy()
                self.B_contacts_des[leg] = self.B_contacts[leg].copy()

        #on loop
        #print(f"standUp leg...{time}")
        alpha = sm.timer.get_elapsed_time(time)/sm.get_state_duration()
        #update des feet positions Z component
        for leg in range(4):
            self.B_contacts_des[leg][2] = (1-alpha) *self.B_contacts_sampled[leg][2] + alpha * (-self.robot_height)

        #compute ik
        for leg in range(4):
            q_des_leg, isFeasible = self.IK.ik_leg(self.B_contacts_des[leg],
                                                   self.leg_names[leg],
                                                   self.legConfig[self.leg_names[leg]][0],
                                                   self.legConfig[self.leg_names[leg]][1])
            if isFeasible:
                self.u.setLegJointState(leg, q_des_leg, self.q_des)
        self.tau_ffwd, self.grForcesW_des = self.wbc.gravityCompensationBase(self.B_contacts,  self.wJ, self.h_joints,  self.basePoseW)
        #self.visualizeContacts()
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
        #termination
        if sm.timer.is_elapsed(time):
            print("standUp Accomplished → next state")
            sm.next(time)

    def startupProcedure(self):
        #ros.sleep(.5)
        print(colored("Starting up", "blue"))
        if self.robot_name == 'hyq' or self.robot_name == 'solo':
            super(QuadrupedController, self).startupProcedure()
            return
        self.q_des = self.q.copy()
        self.pid = PidManager(self.joint_names)
        self.pid.setPDjoints(self.kp_j, self.kd_j, self.ki_j)

        if self.go0_conf == 'standUp':
            self._startup_from_stand_up()
        elif self.go0_conf == 'standDown':
            # this is needed to avoid initial instability in the real robot
            for i in range(10):
                self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
                ros.sleep(0.01)
            self.homing_sm = StateMachine(verbose=0)
            # IMU BIAS ESTIMATION
            if self.real_robot and (self.robot_name == 'go1' or self.robot_name == 'go2' or self.robot_name == 'aliengo'):
                self.homing_sm.add_state("IMUBiasEstimation", self.IMUBiasEstimation, 1.)
            self.homing_sm.add_state("goFold", self.goFoldConfig, 1.)
            self.homing_sm.add_state("searchingContacts", self.searchingContacts)
            self.homing_sm.add_state("applyGravityComp", self.applyGravityComp, 1.)
            self.homing_sm.add_state("standUp", self.standUp, 2.)
            self.homing_sm.start(start_time=self.time)
            try:
                while not ros.is_shutdown() and self.homing_sm.running:
                    self.updateKinematics()
                    self.homing_sm.step(self.time)
                    self.rate.sleep()
                    self.time = np.round(self.time + self.dt, 4)  # np.array([self.loop_time]), 3)
            except (ros.ROSInterruptException, ros.service.ServiceException):
                ros.signal_shutdown("killed")
                self.deregister_node()

    def _startup_from_stand_up(self):
        for i in range(10):
            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            ros.sleep(0.01)
        self.q_des = conf.robot_params[self.robot_name]['q_0']
        alpha = 0.
        try:
            print(colored(f"[startupProcedure to {self.q_des} t: " + str(self.time[0]) + "s] applying gravity compensation", "blue"))
            GCStartTime = self.time
            while not ros.is_shutdown():
                q_norm = np.linalg.norm(self.q - self.q_des)
                qd_norm = np.linalg.norm(self.qd - self.qd_des)
                if q_norm < 0.1 and qd_norm < 0.1 or self.time > 5:
                    break
                self.updateKinematics()
                # self.visualizeContacts()
                GCTime = self.time - GCStartTime
                if GCTime <= self.gravity_comp_duration:
                    if alpha < 1:
                        alpha = GCTime/self.gravity_comp_duration

                self.send_command(self.q_des, self.qd_des, alpha*p.wbc.gravityCompensationBase(self.B_contacts,
                                                            self.wJ,
                                                            self.h_joints,
                                                            self.basePoseW))

            # IMU BIAS ESTIMATION
            if self.real_robot and (self.robot_name == 'go1' or self.robot_name == 'go2' or self.robot_name == 'aliengo'):
                print(colored("[startupProcedure t: " + str(self.time[0]) + "s] Imu bias estimation", "blue"))
                # print('counter: ' + self.imu_utils.counter + ', timeout: ' + self.imu_utils.timeout)
                while self.imu_utils.counter < self.imu_utils.timeout:
                    self.updateKinematics()
                    self.imu_utils.IMU_bias_estimation(self.b_R_w, self.baseLinAccB)
                    self.tau_ffwd[:] = 0.
                    self.send_command(self.q_des, self.qd_des, self.tau_ffwd)


        except (ros.ROSInterruptException, ros.service.ServiceException):
            ros.signal_shutdown("killed")
            self.deregister_node()


    def gracefulCollapse(self):
        self.alphaCollapse -= self.dt / self.collapseTime
        if self.alphaCollapse<=0:
            self.alphaCollapse = 0.0
            if self.state_estimation=='pronto':
                os.system(" rosnode kill /aliengo_joint_swapper")
                os.system(" rosnode kill /pronto_aliengo")
            ros.signal_shutdown("killed")
            self.deregister_node()
            return True
        else:
            self.pid.setPDjoints(self.alphaCollapse * self.kp_act, self.alphaCollapse * self.kd_act, self.alphaCollapse * self.ki_act)
            return False

    def getCoMReference(self, com_conf, robot_height, com_initial_pos_xy, com_initial_vel_xy):
        # p_init = [px0, py0, px1, py1] = initial position of the two feet on the ground
        # hip_pos: dictionary (keyed by ee_frame name, e.g. "lf_foot") containing XY pos of the hips w.r.t. the CoM
        # gait_pattern: list containing names of support feet for every time step

        # use the actual foot placement (base frame), not the HAA joint position: the HAA joint
        # is ~0.083 m medial to the standing foot (the hfe joint's lateral offset), so anchoring
        # footholds to it pulls every foothold inward toward the centerline
        hip_pos = {foot_frame: self.B_contacts[i][:2].copy() for i, foot_frame in enumerate(com_conf.ee_frames)}

        # MPC Parameters:
        nb_dt_per_step = int(round(com_conf.T_step / com_conf.dt_mpc))
        N = com_conf.nb_steps * nb_dt_per_step  # number of desired walking intervals

        # CoM initial state:
        x_0 = np.array([com_initial_pos_xy[0], com_initial_pos_xy[1], com_initial_vel_xy[0], com_initial_vel_xy[1]])

        #initial position of the feet (decide with which feet to start)
        p_0 = np.concatenate([x_0[:2] + hip_pos["lf_foot"], x_0[:2] + hip_pos["rh_foot"]])

        # compute Com reference trajectories:
        C_ref = np.zeros((2, N + 1))  # not used
        DC_ref = np.tile(com_conf.step_length / com_conf.T_step, N + 1)
        DC_ref = np.vstack([DC_ref, np.zeros(N + 1)])
        gait_pattern = []
        while (len(gait_pattern) < N + 1):
            gait_pattern += nb_dt_per_step * [["lf_foot", "rh_foot"]]
            gait_pattern += nb_dt_per_step * [["rf_foot", "lh_foot"]]

        ocp = SrbFootstepOcp(com_conf.dt_mpc, N, robot_height)


        start = time.time()
        sol = ocp.solve(x_0, p_0, com_conf.wc, com_conf.wdc, com_conf.wu, com_conf.wp, C_ref, DC_ref, hip_pos, gait_pattern)
        print("Computation time", time.time() - start)

        com_state, u, p = sol.value(ocp.x), sol.value(ocp.u), sol.value(ocp.p)
        foot_steps, cop, hip_pos_0, hip_pos_1 = np.zeros((4, N)), np.zeros((2, N)), np.zeros((2, N)), np.zeros((2, N))
        j = 0
        for i in range(N):
            if (i > 0 and gait_pattern[i][0] != gait_pattern[i - 1][0]):
                j += 1
            foot_steps[:, i] = sol.value(ocp.p[:, j])
            cop[:, i] = foot_steps[:2, i] + u[i] * (foot_steps[2:, i] - foot_steps[:2, i])
            hip_pos_0[:, i] = com_state[:2, i] + hip_pos[gait_pattern[i][0]]
            hip_pos_1[:, i] = com_state[:2, i] + hip_pos[gait_pattern[i][1]]

        return com_state, foot_steps, cop,  gait_pattern

    def generateInterpolatedReference(self, com_conf, com_state, foot_steps, cop, gait_pattern, robot_height):
        # INTERPOLATE WITH TIME STEP OF CONTROLLER
        dt_ctrl = com_conf.dt  # time step used by controller
        com_state_x = com_state[[0, 2], :]
        com_state_y = com_state[[1, 3], :]
        cop_x = cop[0, :]
        cop_y = cop[1, :]
        com, dcom, ddcom, cop, foot_steps_ctrl = interpolate_lipm_traj(
            com_conf.T_step, com_conf.nb_steps, com_conf.dt_mpc, dt_ctrl,
            robot_height, com_conf.g,
            com_state_x, com_state_y, foot_steps, cop_x, cop_y)

        # COMPUTE TRAJECTORIES FOR FEET
        N = com_state.shape[1] - 1  # number of time steps for traj-opt
        N_ctrl = int((N * com_conf.dt_mpc) / dt_ctrl)  # number of time steps for control
        x, dx, ddx = {}, {}, {}
        for foot_name in com_conf.ee_frames:
            if (foot_name in gait_pattern[0]):
                shift = 0
                initial_phase = "stance"
                gp = gait_pattern[0]
            else:
                shift = 1
                initial_phase = "swing"
                i = 1
                while (gait_pattern[i][0] == gait_pattern[i - 1][0]):
                    i += 1
                gp = gait_pattern[i]
            if (foot_name == gp[0]):
                indices = [0, 1]
            else:
                indices = [2, 3]

            nb_dt_per_step = int(round(com_conf.T_step / com_conf.dt_mpc))
            target_foot_steps = foot_steps[indices, shift::2*nb_dt_per_step]
            x[foot_name], dx[foot_name], ddx[foot_name] = compute_foot_traj(target_foot_steps, N_ctrl, dt_ctrl, com_conf.T_step, com_conf.step_height, initial_phase)

        return com,  dcom, ddcom, x, dx, ddx, cop


    def plotReference(self):
        if conf.plotting:
            N = foot_steps.shape[1]
            plt.figure()
            plt.plot(com_state[0, :N], color='green', label="CoM X pos")

            plt.plot(cop[0, :], color='blue',label="CoP X")
            plt.plot(foot_steps[0, :], ':', color='black', label="foot steps 0 X")
            plt.plot(foot_steps[2, :], ':', color='black', label="foot steps 1 X")
            plt.grid(True)
            plt.legend()

            # plt.figure()
            # plt.plot(com_state[2, :N], color='green', label="CoM X vel")
            # plt.grid(True)
            # plt.legend()
            plt.figure()
            plt.plot(com_state[1, :N], color='green', label="CoM Y pos")
            plt.plot(cop[1, :], color='blue', label="CoP Y")
            plt.plot(foot_steps[1, :], ':',color='black',label="foot steps 0 Y")
            plt.plot(foot_steps[3, :], ':', color='black', label="foot steps 1 Y")
            plt.legend()
            plt.grid(True)
            #
            # plt.figure()
            # plt.plot(com_state[2, :N], color='green', label="CoM Y vel")
            # plt.grid(True)
            # plt.legend()
            # plt.show( )

            plt.figure()
            plt.plot(com_state[0, :N], com_state[1, :N], color='green', label="CoM XY")
            plt.plot(cop[0, :], cop[1, :], color='blue', label="CoP XY")
            plt.scatter(foot_steps[0, :], foot_steps[1, :], facecolors='none', edgecolors='black', label="footholds")
            plt.scatter(foot_steps[2, :], foot_steps[3, :], facecolors='red', edgecolors='black')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.axis('equal')
            plt.legend()
            plt.grid(True)
            plt.pause(0.001)

if __name__ == '__main__':
    p = QuadrupedController('aliengo')
    world_name = 'fast.world'
    use_gui = False
    p.state_estimation = 'ground_truth' # 'odometry',  'pronto', 'ground_truth' (only sim), 'mocap'
    rl_control = 'state_est_based' #'none',  'state_est_based'
    # NOTE: in the RL controller, SE NN is used only if state estimation is not pronto
    rl_use_nn_se = p.state_estimation != 'pronto'
    p.controller_type = 'tsid' # 'quasi-static', 'tsid'

    use_joy = True
    generate_reference = False
    p.SAVE_BAG = False  #
    if p.robot_name == 'go2':
        p.custom_locosim_launch_file = True

    if rl_control == 'state_est_based':
        rl_controller = RlVelocityController(p.robot_name, p.dt, use_nn_se=rl_use_nn_se, debug=True)

    try:
        p.startController(world_name=world_name,
                          use_ground_truth_contacts=True,
                          additional_args=['gui:='+str(use_gui),
                                           'go0_conf:=standDown',
                                           'rviz:=true',
                                           *(['task_period:=0.002'] if p.real_robot else [])]) #change task period to 500Hz instead of 1000Hz for real robot
        if p.SAVE_BAG:
            now = datetime.now()
            format_date = now.strftime("%Y-%m-%d-%H-%M-%S")
            p.recorder = RosbagControlledRecorder(
                topics='/aliengo/joint_states /aliengo/trunk_imu /aliengo/ground_truth /rl_ref_vel /tf /tf_static',
                        bag_name="test_" + format_date + ".bag", record_from_startup_=False)
            p.recorder.start_recording_srv()
        if use_joy:
            joy = JoyManager("js1", end_scale = 0.2)

        # for quick startup
        #p.resetRobot(basePoseDes=np.array([0.0, -0.0,  0.356, -0.0, -0.0, 0.0]))
        p.startupProcedure()
        # to reduce simulation frequency
        #p.setSimSpeed(dt_sim=0.001, max_update_rate=100, iters=1500)

        if p.state_estimation=='pronto':
            launchFileNode("mocap_qualisys", "qualisys.launch")
            launchFileNode("pronto_aliengo", "pronto_aliengo.launch", additional_args=['pronto_conf:='+p.pronto_config,
                                                                                       'use_sim_time:='+str(not p.real_robot)])
        if rl_control != 'none':
            p.pid.setPDjoints(rl_controller.kp, rl_controller.kd, np.full(12,0))
        else:
            if p.controller_type == 'quasi-static':
                # softer "wbc" gains since tau_ffwd already supplies most of the required torque;
                # the same PD tracks q_des/qd_des on every joint, computed below by Wcom2Joints_des()
                # for both stance legs (whole-body IK, feet held planted) and swing legs (foot-trajectory IK)
                p.pid.setPDjoints(p.kp_wbc_j, p.kd_wbc_j, p.ki_wbc_j)
            if p.controller_type == 'tsid':
                # TSID whole-body controller (kept for reference / easy switch-back, see the matching
                tsid_quadruped = TsidQuadruped(com_optim_conf, p.configuration.copy(), p.gen_velocities.copy())
                # with TSID (pure feedforward torque from the QP), the low-level joint PD must be off instead:
                p.pid.setPDs(0, 0, 0)

        p.startTime = p.time
        if generate_reference:
            p.ref_gen = QuadrupedTasks(task='pushup', robot_conf=conf.robot_params[p.robot_name], gui=True, quadruped=p)
            p.ref_gen.startUp(p.time)


        #compute robot reference
        com_state, foot_steps, cop,  gait_pattern = p.getCoMReference(com_optim_conf, p.robot_height, p.comPoseW[:2], p.comTwistW[:2])
        com_ref,  dcom_ref, ddcom_ref, x_ref, dx_ref, ddx_ref, cop_ref = p.generateInterpolatedReference(com_optim_conf, com_state, foot_steps, cop,   gait_pattern, p.robot_height)

        # plot references
        p.plotReference()

        # quasi-static CoM controller (p.wbc.computeWBC) + per-leg joint PD running in parallel,
        # initialized with the current (standing) robot state
        p.updateKinematics()

        foot_swing_thresh = 1e-4  # foot z-reference above this -> foot is swinging

        counter = 0
        # Wbase2Joints_des()/Wcom2Joints_des() integrate q_des open-loop between IK solves, so
        # prime them with the actual state before the first call (see its docstring)
        p.q_des = p.q.copy()
        p.qd_des = np.zeros(p.robot.na)
        # no orientation/angular-velocity reference from the (2d) LIPM plan: keep the trunk level
        p.comPoseW_des[3:] = 0.
        p.comTwistW_des[3:] = 0.
        p.comAccW_des[:] = 0.
        # robot starts standing on all 4 feet: latch their current position as the stance target
        p.stance_legs[:] = True
        for leg in range(4):
            p.W_contacts_des[leg] = p.W_contacts[leg].copy()


        print(colored(f"Starting main loop  T =  {p.time}", "blue"))
        while not ros.is_shutdown():
            p.updateKinematics()
            if p.gracefulCollapseFlag:
                if p.gracefulCollapse():
                    break
            if use_joy:
                long_x, long_y, rot_z, buttons = joy.getVelocityReferences()
                # safety layer
                if buttons[0] and not p.gracefulCollapseFlag:
                    print(colored("start Graceful collapse", "red"))
                    p.kp_act, p.kd_act, p.ki_act = p.pid.getPDjoints()
                    print(colored(f"Storing actual pd: {p.kp_act}, {p.kd_act},{p.ki_act}"), "red")
                    p.gracefulCollapseFlag = True
                if buttons[1]:
                    print(colored("Severe shutdown!", "red"))
                    if p.state_estimation == 'pronto':
                        os.system(" rosnode kill /aliengo_joint_swapper")
                        os.system(" rosnode kill /pronto_aliengo")
                    ros.signal_shutdown("killed")
                    p.deregister_node()
                    break

            #p.applyForce(0, 100, 0, 0, 0, 0, 0.25)
            if rl_control != 'none':
                if use_joy:
                    rl_controller.velocity_cmd = np.array([long_x, long_y, rot_z])
                else:
                    rl_controller.velocity_cmd = np.random.uniform(low=-0.4, high=0.4, size=(3,))
                        
                p.baseTwistW_des[:3] = p.b_R_w.T @ np.append(rl_controller.velocity_cmd[:2], 0.0)
                p.baseTwistW_des[5] = rl_controller.velocity_cmd[2]
                if rl_control == 'state_est_based':
                    # Compute observations for the policy
                    # Disable the lin_vel_b and use lin_acc_b observation if using NN SE
                    if rl_controller.use_nn_se and not rl_controller.debug:
                        lin_acc_b = p.baseLinAccB
                        lin_vel_b = None
                    elif rl_controller.use_nn_se and rl_controller.debug:
                        lin_acc_b = p.baseLinAccB
                        lin_vel_b = p.b_R_w.dot(p.baseTwistW[:3])
                    else:
                        lin_acc_b = None
                        lin_vel_b = p.b_R_w.dot(p.baseTwistW[:3])
                    ang_vel_b = p.b_R_w.dot(p.baseTwistW[3:6])
                    proj_gravity = p.b_R_w.dot(np.array([0,0,-1]))

                    p.rl_q_des = rl_controller.action(lin_acc_b, lin_vel_b, ang_vel_b, proj_gravity, p.q, p.qd, policy_type="default")

                #switch off wbc
                p.grForcesW_des = np.zeros((12))
                p.tau_ffwd = np.zeros(12)
                p.send_command(p.rl_q_des, np.zeros(12), np.zeros(12), log_data_in_send_command=True)
            else:

                if generate_reference:
                    p.q_des, p.qd_des, p.tau_ffwd, p.basePoseW_des, p.baseTwistW_des = p.ref_gen.generateReference(p.time)
                else:
                    ####################
                    # get reference from optimization trajectory interpolated
                    #########################
                    idx = min(counter, com_ref.shape[1] - 1)
                    p.comPoseW_des[:3] = com_ref[:, idx]
                    p.comTwistW_des[:3] = dcom_ref[:, idx]
                    p.comAccW_des[:3] = ddcom_ref[:, idx]


                    # cop_ref
                    #plot desired cop
                    p.cop_des = cop_ref[:, idx]
                    p.ros_pub.add_marker(np.concatenate((p.cop_des, [0])), radius=0.1, color="red")
                    # plot actual cop
                    p.cop_act = p.computeCoP()
                    p.ros_pub.add_marker(np.concatenate((p.cop_act,  [0])), radius=0.1, color="blue")

                    if counter < com_ref.shape[1] - 1:
                        counter += 1

                    if p.controller_type == 'tsid':
                        ####################################
                        # TSID whole-body controller (CoM task + swing-foot tasks + point contacts)
                        # kept for reference / easy switch-back: uncomment this block (and the
                        # tsid_quadruped instantiation above) and comment out the quasi-static block
                        # below to use TSID again instead
                        ####################################
                        tsid_quadruped.set_com_ref(com_ref[:, idx], dcom_ref[:, idx], ddcom_ref[:, idx])

                        for foot_name in com_optim_conf.ee_frames:
                            leg = p.u.leg_map[foot_name[:2].upper()]
                            # feed the planned foot trajectory into the des-foot log (plotContacts), since
                            # this walking loop never otherwise touches p.W_contacts_des during locomotion
                            p.W_contacts_des[leg] = x_ref[foot_name][:, idx]
                            #planned liftoff
                            is_swinging = x_ref[foot_name][2, idx] > foot_swing_thresh
                            #haptic TD
                            # if is_swinging:
                            #     p.stance_legs[leg] = False
                            #     tsid_quadruped.remove_contact(foot_name, transition_time=com_optim_conf.contact_transition_time)
                            # # haptic touchdown
                            # elif p.contact_state[leg]:
                            #     # plan says stance, but only rigidify the contact once the
                            #     # foot is actually sensed on the ground (avoids commanding a
                            #     # ground reaction force at a foot that hasn't touched down yet)
                            #     p.stance_legs[leg] = True
                            #     tsid_quadruped.add_contact(foot_name)

                            # non haptic TD: plan says stance, but only latch the (fixed)
                            if is_swinging:
                                p.stance_legs[leg] = False
                                # swing leg: track the planned foot trajectory (world frame)
                                tsid_quadruped.remove_contact(foot_name,  transition_time=com_optim_conf.contact_transition_time)
                            else:
                                # plan says stance but touchdown not sensed yet: keep reaching for the
                                # last commanded (swing) target instead of freezing the leg mid-air
                                p.stance_legs[leg] = True
                                tsid_quadruped.add_contact(foot_name)

                            tsid_quadruped.set_foot_3d_ref(foot_name, x_ref[foot_name][:, idx],
                                                            dx_ref[foot_name][:, idx], ddx_ref[foot_name][:, idx])

                        HQPData = tsid_quadruped.compute_problem(float(p.time), p.configuration, p.gen_velocities)
                        sol = tsid_quadruped.solve(HQPData)
                        if sol.status != 0:
                            print(colored(f"QP problem could not be solved! Error code: {sol.status}", "red"))
                        else:
                            p.tau_ffwd = tsid_quadruped.get_torques(sol)
                            for foot_name in com_optim_conf.ee_frames:
                                leg = p.u.leg_map[foot_name[:2].upper()]
                                p.u.setLegJointState(leg, tsid_quadruped.get_contact_force(foot_name, sol), p.grForcesW_des)

                    else:
                        #####################################
                        # quasi-static CoM controller: stance legs are tracked in parallel by a joint PD
                        # (q_des/qd_des from the whole-body IK below) plus p.wbc.computeWBC's ffwd torque;
                        # swing legs are pure joint-PD tracking of the planned foot trajectory (computeWBC
                        # zeroes out their columns, so they only get the constant h_joints bias, not a real ffwd)
                        #####################################
                        for foot_name in com_optim_conf.ee_frames:
                            leg = p.u.leg_map[foot_name[:2].upper()]
                            # planned liftoff
                            is_swinging = x_ref[foot_name][2, idx] > foot_swing_thresh
                            #haptic TD
                            # if is_swinging:
                            #     p.stance_legs[leg] = False
                            #     # swing leg: track the planned foot trajectory (world frame)
                            #     p.W_contacts_des[leg] = x_ref[foot_name][:, idx]
                            # elif p.contact_state[leg]:
                            #     # haptic touchdown: plan says stance, but only latch the (fixed) stance
                            #     # target once the foot is actually sensed on the ground, and only once,
                            #     # otherwise it would just track the (possibly slipping) actual position
                            #     if not p.stance_legs[leg]:
                            #         p.W_contacts_des[leg] = p.W_contacts[leg].copy()
                            #     p.stance_legs[leg] = True
                            # else:
                            #     # plan says stance but touchdown not sensed yet: keep reaching for the
                            #     # last commanded (swing) target instead of freezing the leg mid-air
                            #     p.stance_legs[leg] = False

                            #non haptic TD: plan says stance, but only latch the (fixed)
                            if is_swinging:
                                p.stance_legs[leg] = False
                                # swing leg: track the planned foot trajectory (world frame)
                                p.W_contacts_des[leg] = x_ref[foot_name][:, idx]
                            else:
                                # plan says stance but touchdown not sensed yet: keep reaching for the
                                # last commanded (swing) target instead of freezing the leg mid-air
                                p.stance_legs[leg] = True

                        # map CoM pose/twist + per-foot targets above into q_des/qd_des for all 12 joints
                        p.Wcom2Joints_des()

                        p.tau_ffwd, p.grForcesW_des = p.wbc.computeWBC(p.W_contacts, p.wJ, p.h_joints, p.basePoseW,
                                                                     p.comPoseW, p.baseTwistW, p.comTwistW,
                                                                     p.comPoseW_des, p.comTwistW_des, p.comAccW_des,
                                                                     p.centroidalInertiaB,
                                                                     comControlled=True, type='projection',
                                                                     stance_legs=p.stance_legs)

                p.send_command(p.q_des, p.qd_des, p.alphaCollapse*p.tau_ffwd, log_data_in_send_command=True)

            p.visualizeContacts()
        
    except (ros.ROSInterruptException, ros.service.ServiceException):
        if p.SAVE_BAG:
            p.recorder.stop_recording_srv()
        ros.signal_shutdown("killed")
        p.deregister_node()
    except Exception:
        # don't let an unexpected crash (e.g. TSID/pinocchio failing on a
        # degenerate configuration when the robot falls) skip the final plots
        traceback.print_exc()

    if conf.plotting:
        plotJoint('position', time_log=p.time_log, q_log=p.q_log, q_des_log=p.q_des_log, sharex=True, sharey=False,
                  start=0, end=-1)
        #plotJoint('torque', time_log=p.time_log, tau_des_log=p.tau_ffwd_log)
        plotFrame('position', time_log=p.time_log, des_Pose_log=p.comPoseW_des_log, Pose_log=p.comPoseW_log,
                  title='CoM', frame='W', sharex=True, sharey=False, start=0, end=-1)
        plotFrame('velocity', time_log=p.time_log, des_Twist_log=p.comTwistW_des_log, Twist_log=p.comTwistW_log,
                  title='CoM', frame='W', sharex=True, sharey=False, start=0, end=-1)
        plotContacts('position', time_log=p.time_log, des_LinPose_log=p.W_contacts_des_log, LinPose_log=p.W_contacts_log,
                      frame='W', title='Feet position and contact state')

        fig = plt.figure()
        fig.suptitle('CoM and CoP XY tracking', fontsize=20)
        plt.plot(p.comPoseW_des_log[0, :], p.comPoseW_des_log[1, :], color='green', lw=2, label='CoM des')
        plt.plot(p.comPoseW_log[0, :], p.comPoseW_log[1, :], color='green', linestyle='-',  lw=1, label='CoM act')
        plt.plot(p.cop_des_log[0, :], p.cop_des_log[1, :], color='blue', lw=2, label='CoP des')
        plt.plot(p.cop_log[0, :], p.cop_log[1, :], color='blue', linestyle='-', lw=1, label='CoP act')
        plt.scatter(foot_steps[0, :], foot_steps[1, :], facecolors='none', edgecolors='black', label='footholds')
        plt.scatter(foot_steps[2, :], foot_steps[3, :], facecolors='none', edgecolors='black')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)

        fig = plt.figure()
        fig.suptitle('Stance legs (planned)', fontsize=20)
        leg_names = ['LF', 'LH', 'RF', 'RH']
        for leg in range(4):
            ax = plt.subplot(4, 1, leg + 1, sharex=fig.axes[0] if leg > 0 else None)
            plt.plot(p.time_log, p.stance_legs_log[leg, :], color='black', lw=1)
            plt.ylabel(leg_names[leg])
            plt.ylim([-0.2, 1.2])
            plt.yticks([0, 1], ['swing', 'stance'])
            plt.grid(True)
        plt.xlabel('Time [s]')

        plt.ion()
        plt.show()



    if p.SAVE_BAG:
        p.recorder.stop_recording_srv()
