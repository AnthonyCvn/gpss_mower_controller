#!/usr/bin/env python

# ROS libraries for Python.
import rospy

# ROS messages.
from geometry_msgs.msg import Twist

# Python packages.
import numpy as np
from scipy import linalg
from math import sin, cos

# Specific controller's libraries.
from toolbox import wraptopi
from tf_mng import TfMng
from linear_mpc_controller import Controller


class Filter:
    """ Filter that merge odometry and photogrametry sensors at a given sampling frequency.

    The sensor fusion is performed by an Extended Kalman Filter (EKF).

    Attributes:
        Ts              : Sampling period
        sensor_delay    : Delay between the sensors timestamp and the begining of the sampling period.
        tf_mng          : Transform manager that store sensors value relative to /world coordinate.
        ctrl            : Controller that compute the input command.
        dim_x           : Dimension of vector x.
        dim_u           : Dimension of vector u.
        dim_z           : Dimension of vector z.
        u               : Input command vector.
        z               : Sensors vector.
        mu              : Estimate of the pose of the robot.
        G               : Jacobian g'(x) = dg(x)/dx of the motion function; x = g(u, x) .
        S               : Measurement covariance matrix.
        H               : Jacobian h'(x) = dh(x)/dx of the output function; z = h(x).
        Q               : Measurement covariance matrix.
        R               : State transition covariance matrix.
        cmd_vel         : Twist message of the velocity command.
        pub_cmd         : ROS publisher of cmd_vel.
    """
    def __init__(self):
        """ ... """
        self.Ts = 0.1

        self.sensor_delay = 0.0

        self.tf_mng = TfMng()
        self.ctrl = Controller()

        self.dim_x = 3
        self.dim_u = 2
        self.dim_z = 6

        self.u = np.zeros((self.dim_u, 1))
        self.z = np.zeros((self.dim_z, 1))
        self.mu = np.zeros((self.dim_x, 1))

        self.G = np.eye(self.dim_x)
        self.S = 0.1 * np.eye(self.dim_x)
        self.H = np.vstack((np.eye(self.dim_x), np.eye(self.dim_x)))

        # Covariance matrices (Q: measurement (odometry then markers), R: state transition).
        self.Q = np.diag(np.array([1.0e-12, 1.0e-12, 1.0e-12, 5.0e-3, 5.0e-3, 5.0e-3]))
        self.R = 1.0e-1 * np.eye(self.dim_x)

        self.cmd_vel = Twist()
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def run(self):
        """ Start the filter and initialize a timer at the given sampling frequency. """
        rospy.loginfo("Sampling frequency set at {0} Hz".format(1.0/self.Ts))

        self.tf_mng.run()

        if self.tf_mng.photo_activated:
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_with_marker)
        else:
            self.z[0:3] = self.tf_mng.sensors.odom_pose
            self.mu = self.z[0:3]
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_odom_based)

    def timer_cb_odom_based(self, event):
        """ Timer callback running at sampling frequency (based only on odometry).

        For every sampling period:
        1)  Store the sensors information from the transformer manager.
        2)  Estimate the pose of the robot with the Extended Kalman Filter.
        3)  Delay compensation of the robot's pose with the motion model.
        4)  Send the velocity command.
        5)  Update the pose of the robot in the transform manager.

        """
        self.z[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot

        self.sensor_delay = event.current_real.to_sec() - self.tf_mng.sensors.t

        if self.sensor_delay > self.Ts/2:
            rospy.loginfo("!! Large sensor's delay of {0} ms !!".format(self.sensor_delay*1000))

        self.ekf()

        mu_corr = self.motion_model(self.u, self.mu, self.sensor_delay)

        self.u = self.ctrl.compute(self.mu) # mu_corr if clock are synchronized

        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)


    def timer_cb_with_marker(self, event):
        """ Timer callback running at sampling frequency (based on odometry and photogramtery) """

    def motion_model(self, u, x, dt):
        """ Motion model of the mobile robot.

        Args:
            u   : Input vector (v, w).
            x   : State vector (x, y, phi).
            dt  : Delta time.

        Returns:
            g   : Predicted state after time dt starting from state x.

        """
        g = np.zeros((self.dim_x, 1))

        g[0] = x[0] + dt * u[0] * 1.0 * cos(x[2] + u[1] * dt)
        g[1] = x[1] + dt * u[0] * 1.0 * sin(x[2] + u[1] * dt)
        g[2] = wraptopi(x[2] + dt * u[1])

        return g

    def jacobian_motion_model(self, u, x, dt):
        """ Jacobian of the motion model.

        Args:
            u   : Input vector (v, w).
            x   : State vector (x, y, phi).
            dt  : Sampling period.

        Returns:
            G   : Jacobian of the motion model.

        """
        G = np.eye(self.dim_x)

        G[0, 2] = - dt * u[0] * 1.0 * sin(x[2] + u[1] * dt)
        G[1, 2] = + dt * u[0] * 1.0 * cos(x[2] + u[1] * dt)

        return G

    def measure_subtraction(self, z1, z2):
        """ Substraction of two measurement z1 and z2.
        Args:
            z1      : First measurement vector (x_odom, y_odom, phi_odom, x_photo, y_photo, phi_photo).
            z2      : Second Measurement vector (x_odom, y_odom, phi_odom, x_photo, y_photo, phi_photo).

        Returns:
            delta   : (x, y, phi) =  z1 - z2 with the angles phi wrapped between -pi and pi.
        """
        delta = z1 - z2
        delta[2] = wraptopi(delta[2])
        delta[5] = wraptopi(delta[5])

        return delta

    def ekf(self):
        """ Extended Kalman Filter (EKF).

        The EKF proceed as following:
        1) Prediction phase.
        2) Optimal Kalman gain computation.
        3) Measurement update.

        """
        # 1) Prediction phase.
        self.G = self.jacobian_motion_model(self.u, self.mu, self.Ts)
        self.mu = self.motion_model(self.u, self.mu, self.Ts)
        self.S = self.G.dot(self.S).dot(self.G.T) + self.R

        # 2) Optimal Kalman gain
        optimal_gain = self.S.dot(self.H.T).dot(linalg.inv(self.H.dot(self.S).dot(self.H.T) + self.Q))

        # 3) ControllerManagerMeasurement update
        self.mu = self.mu + optimal_gain.dot(self.measure_subtraction(self.z, np.vstack((self.mu, self.mu))))
        self.S = (np.eye(self.dim_x) - optimal_gain.dot(self.H)).dot(self.S)

    def twist_pose_pub(self, numpy_pose, twist_publisher):
        """ Publish a numpy array that describe a pose (x, y, phi) on a Twist ROS message. """
        twist_pose = Twist()
        twist_pose.linear.x = numpy_pose[0]
        twist_pose.linear.y = numpy_pose[1]
        twist_pose.angular.z = numpy_pose[2]
        twist_publisher.publish(twist_pose)
