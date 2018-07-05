#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import numpy as np
from scipy import linalg
from math import sin, cos

from toolbox import wraptopi
from tf_mng import TfMng

from linear_mpc_controller import Controller


class Filter:
    """Filter that merge odometry and fiducial markers information at a given sampling time.

    ...

    Attributes:
        ...: ...
    """
    def __init__(self):
        """ ... """
        self.Ts = 0.1

        self.tf_mng = TfMng()
        self.ctrl = Controller()

        self.dim_x = 3
        self.dim_u = 2
        self.dim_z = 6

        self.u = np.zeros((self.dim_u, 1))
        self.z = np.zeros((self.dim_z, 1))
        self.mu = np.zeros((self.dim_x, 1))

        self.G = np.eye(self.dim_x)

        # Variance and measurement matrix.
        self.S = 0.1 * np.eye(self.dim_x)
        self.H = np.vstack((np.eye(self.dim_x), np.eye(self.dim_x)))

        # Covariance matrices (Q: measurement (odometry then markers), R: state transition).
        self.Q = np.diag(np.array([1.0e-12, 1.0e-12, 1.0e-12, 5.0e-3, 5.0e-3, 5.0e-3]))
        self.R = 1.0e-1 * np.eye(self.dim_x)

        # States publisher
        self.publish_states = True
        self.pub_global_robot_pose = rospy.Publisher('/robot0/pose_estimate', Twist, queue_size=1)
        self.pub_global_odom_pose = rospy.Publisher('/robot0/global_odom_pose', Twist, queue_size=1)
        self.pub_global_marker_pose = rospy.Publisher('/robot0/global_marker_pose', Twist, queue_size=1)

        # Command publisher
        self.cmd_vel = Twist()
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def run(self):
        """ ... """
        rospy.loginfo("Sampling frequency set at {0} Hz".format(1.0/self.Ts))

        self.tf_mng.run()

        if self.tf_mng.fid_marker_activated:
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_with_marker)
        else:
            self.z[0:3] = self.tf_mng.sensors.odom_pose
            self.mu = self.z[0:3]
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_odom_based)

    def timer_cb_odom_based(self, event):
        """ ... """
        self.z[0:3] = self.tf_mng.sensors.odom_pose
        saved_T_odom2robot = self.tf_mng.T_odom2robot
        # print self.tf_mng.sensors.t
        # print event.current_real.to_sec()
        # print str((event.current_real.to_sec() - self.tf_mng.sensors.t)*1e3)+" ms"
        # print""

        self.ekf()

        self.ctrl.mu = self.mu
        self.ctrl.compute()
        self.u = self.ctrl.u

        # Publish controller command
        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.tf_mng.update_world2odom(self.mu, saved_T_odom2robot)

        if self.publish_states:
            self.twist_pose_pub(self.mu, self.pub_global_robot_pose)
            self.twist_pose_pub(self.z[0:3], self.pub_global_odom_pose)


    def timer_cb_with_marker(self, event):
        """ ... """

    def prediction_model(self, u, x, dt):
        """ ... """
        g = np.zeros((self.dim_x, 1))

        g[0] = x[0] + dt * u[0] * 1.0 * cos(x[2] + u[1] * dt)
        g[1] = x[1] + dt * u[0] * 1.0 * sin(x[2] + u[1] * dt)
        g[2] = wraptopi(x[2] + dt * u[1])

        return g

    def jacobian_prediction_model(self, u, x, dt):
        """ ... """
        G = np.eye(self.dim_x)

        G[0, 2] = - dt * u[0] * 1.0 * sin(x[2] + u[1] * dt)
        G[1, 2] = + dt * u[0] * 1.0 * cos(x[2] + u[1] * dt)

        return G

    def measure_subtraction(self, z1, z2):
        delta = z1 - z2
        delta[2] = wraptopi(delta[2])
        delta[5] = wraptopi(delta[5])
        return delta

    def ekf(self):
        """ ... """
        self.G = self.jacobian_prediction_model(self.u, self.mu, self.Ts)

        # Prediction
        self.mu = self.prediction_model(self.u, self.mu, self.Ts)
        self.S = self.G.dot(self.S).dot(self.G.T) + self.R

        # Optimal Kalman gain
        optimal_gain = self.S.dot(self.H.T).dot(linalg.inv(self.H.dot(self.S).dot(self.H.T) + self.Q))

        # Measurement update
        self.mu = self.mu + optimal_gain.dot(self.measure_subtraction(self.z, np.vstack((self.mu, self.mu))))
        self.S = (np.eye(self.dim_x) - optimal_gain.dot(self.H)).dot(self.S)

    def twist_pose_pub(self, numpy_pose, twist_publisher):
        twist_pose = Twist()
        twist_pose.linear.x = numpy_pose[0]
        twist_pose.linear.y = numpy_pose[1]
        twist_pose.angular.z = numpy_pose[2]
        twist_publisher.publish(twist_pose)
