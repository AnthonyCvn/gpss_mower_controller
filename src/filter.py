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
        Ts                  : Sampling period
        sensor2cont_delay   : Delay between the sensors timestamp and the begining of the sampling period.
        tf_mng              : Transform manager that store sensors value relative to /world coordinate.
        ctrl                : Controller that compute the input command.
        dim_x               : Dimension of vector x.
        dim_u               : Dimension of vector u.
        dim_y               : Dimension of vector z.
        u                   : Input command vector.
        y                   : Sensors vector.
        mu                  : Estimate of the pose of the robot.
        G                   : Jacobian g'(x) = dg(x)/dx of the motion function; x = g(u, x) .
        S                   : Measurement covariance matrix.
        H                   : Jacobian h'(x) = dh(x)/dx of the output function; z = h(x).
        Q                   : Measurement covariance matrix.
        R                   : State transition covariance matrix.
        cmd_vel             : Twist message of the velocity command.
        pub_cmd             : ROS publisher of cmd_vel.
    """
    def __init__(self):
        """ ... """
        self.calibrate_counter = 0
        self.n_calibration_pictures = 10
        self.calibrate = True
        self.is_sample_state_augmentation = True

        self.robot_id = 1
        self.Ts = 0.1

        self.sensor2cont_delay = 0.0

        self.photo_seq = 0

        self.tf_mng = TfMng()
        self.ctrl = Controller()

        self.cmd_vel = Twist()
        self.pub_cmd = None

        self.dim_x = 3
        self.dim_u = 2
        self.dim_y = 6

        self.u = np.zeros((self.dim_u, 1))
        self.y = np.zeros((self.dim_y, 1))
        self.mu = np.zeros((self.dim_x, 1))

        self.G = np.eye(self.dim_x)
        self.H = np.vstack((np.eye(self.dim_x), np.eye(self.dim_x)))
        self.S = 0.1 * np.eye(self.dim_x)

        # Covariance matrices (Q: measurement (odometry then photogrammetry), R: state transition).
        self.Q = np.diag(np.array([1.0e-3, 1.0e-3, 1.0e-3, 5.0e-1, 5.0e-1, 5.0e-1]))
        self.R = 1.0e-3 * np.eye(self.dim_x)

        # Augmented states algorithm variable
        self.mu_a_pred = np.zeros((2 * self.dim_x, 1))
        self.mu_a = np.zeros((2*self.dim_x, 1))
        self.y_a = np.zeros((self.dim_y, 1))

        self.G_a = np.eye(2*self.dim_x)
        self.H_a = np.eye(2*self.dim_x)

        #self.S_a = 100 * np.eye(2*self.dim_x)
        self.S_a = 100 * np.ones((2 * self.dim_x, 2 * self.dim_x))

        # Covariance matrices for augmented states
        # Q_: measurement (odometry then photogrammetry)
        # R_a: state transition (new states then delayed states)
        #self.Q_a = np.diag(np.array([25.0e-4, 25.0e-4, 25.0e-4, 4.0e-6, 4.0e-6, 4.0e-6]))
        #self.R_a = np.diag(np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6, 1.0e-6]))

        self.Q_a = np.diag(np.array([4.0e-4, 4.0e-4, 4.0e-4, 4.0e-6, 4.0e-6, 4.0e-5]))
        self.R_a = np.diag(np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6, 1.0e-5]))

        # Storage of [x, y, phi] from time (k-n_prediction_stored) to time k
        self.prediction_store = []
        for j in range(self.n_calibration_pictures):
            self.prediction_store.append(np.zeros((self.dim_x, 1)))

    def run(self):
        """ Start the filter and initialize a timer at the given sampling frequency. """
        rospy.loginfo("Sampling frequency set at {0} Hz".format(1.0/self.Ts))

        self.tf_mng.run()

        if self.tf_mng.photo_activated:
            if self.is_sample_state_augmentation:
                rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_photo_odom_augmented)
            else:
                rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_photo_odom)
        else:
            self.y[0:3] = self.tf_mng.sensors.odom_pose
            self.mu = self.y[0:3]
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
        self.y[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot

        self.sensor2cont_delay = event.current_real.to_sec() - self.tf_mng.sensors.odom_t

        if self.sensor2cont_delay > 2*self.Ts:
            rospy.loginfo("Robot #{0} has Large sensor-to-controller delay of {1} ms !"
                          .format(self.robot_id, self.sensor2cont_delay*1000))

        # Kalman filter
        self.ekf()

        self.u = self.ctrl.compute(self.mu)

        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)


    def timer_cb_photo_odom_augmented(self, event):
        """ Timer callback running at sampling frequency with augmented states (based on odometry and photogramtery) """
        ### Diego ### self.y[0:3] = self.tf_mng.get_world2robot(self.tf_mng.odom_odom2robot)
        self.y[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot

        # Calibration process
        if self.calibrate:
            if self.tf_mng.sensors.is_photo:
                # Init. the robot's poses (current and stored) with the tag's pose.
                self.mu = self.tf_mng.get_photo_pose(self.tf_mng.photo_world2robot)
                self.y[0:3] = self.mu
                for j in range(self.n_calibration_pictures):
                    self.prediction_store[j] = self.mu
                self.calibrate = False
                rospy.loginfo("Robot #{0} set its pose according to hrp0{0} tag.".format(self.robot_id))
            else:
                return

        ### Diego ### if self.photo_seq is not self.tf_mng.photo_world2robot.header.seq:
        if self.tf_mng.sensors.is_photo:

            ### Diego ### self.photo_seq = self.tf_mng.photo_world2robot.header.seq
            self.tf_mng.sensors.is_photo = False

            # Read sensor value
            self.y[3:6] = self.tf_mng.get_photo_pose(self.tf_mng.photo_world2robot)
            #self.y[3:6] = self.tf_mng.sensors.photo_pose

            # Calculate the delay in number of samples
            self.delay_photo = event.current_real.to_sec() - self.tf_mng.photo_world2robot.header.stamp.to_sec()
            sample_delay = int(round(self.delay_photo/self.Ts))
            if sample_delay >= self.n_calibration_pictures:
                sample_delay = 0

            # Store and recover the past prediction
            self.prediction_store.append(self.mu_a_pred[0:3])
            self.prediction_store.pop(0)
            self.mu_a_pred[3:6] = self.prediction_store[-sample_delay]

            # Case without delay
            if sample_delay == 0:
                self.mu_a_pred[3:6] = self.motion_model(self.u, self.mu, self.Ts)

        self.ekf_augmented()

        self.u = self.ctrl.compute(self.mu)

        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)

    def timer_cb_photo_odom(self, event):
        """ Timer callback running at sampling frequency (based on odometry and photogramtery) """
        self.y[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot

        self.sensor2cont_delay = event.current_real.to_sec() - self.tf_mng.sensors.odom_t
        #print self.sensor2cont_delay*1000

        if self.sensor2cont_delay > 10*self.Ts:
            rospy.loginfo("Robot #{0} has Large sensor-to-controller delay of {1} ms !"
                          .format(self.robot_id, self.sensor2cont_delay*1000))

        if self.tf_mng.sensors.is_photo:
            self.tf_mng.sensors.is_photo = False
            self.H[3, 0] = 1
            self.H[4, 1] = 1
            self.H[5, 2] = 1

            self.y[3:6] = self.tf_mng.sensors.photo_pose

            if self.calibrate:
                self.calibrate = False
                self.mu[0] = self.y[3]
                self.mu[1] = self.y[4]
                self.mu[2] = self.y[5]
                self.y[0] = self.y[3]
                self.y[1] = self.y[4]
                self.y[2] = self.y[5]
        else:
            if self.calibrate:
                return
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0

        # Extended Kalman Filter (EKF)
        self.ekf()

        self.u = self.ctrl.compute(self.mu)

        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)

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

        g[0] = x[0] + dt * u[0] * 1.0 * cos(x[2])
        g[1] = x[1] + dt * u[0] * 1.0 * sin(x[2])
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

        G[0, 2] = - dt * u[0] * 1.0 * sin(x[2])
        G[1, 2] = + dt * u[0] * 1.0 * cos(x[2])

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
        self.mu = self.mu + optimal_gain.dot(self.measure_subtraction(self.y, np.vstack((self.mu, self.mu))))
        self.S = (np.eye(self.dim_x) - optimal_gain.dot(self.H)).dot(self.S)

    def ekf_augmented(self):
        """ Extended Kalman Filter (EKF) for augmented system.
        The EKF proceed as following:
        1) Prediction phase.
        2) Optimal Kalman gain computation.
        3) Measurement update.
        """
        # 1) Prediction phase.
        self.G_a[0:3, 0:3] = self.jacobian_motion_model(self.u, self.mu_a[0:3], self.Ts)
        self.mu_a_pred[0:3] = self.motion_model(self.u, self.mu, self.Ts)
        self.S_a = self.G_a.dot(self.S_a).dot(self.G_a.T) + self.R_a

        # 2) Optimal Kalman gain
        optimal_gain = self.S_a.dot(self.H_a.T).dot(linalg.inv(self.H_a.dot(self.S_a).dot(self.H_a.T) + self.Q_a))

        # 3) ControllerManagerMeasurement update
        self.mu_a = self.mu_a_pred + optimal_gain.dot(self.measure_subtraction(self.y, self.mu_a_pred))
        self.S_a = (np.eye(2 * self.dim_x) - optimal_gain.dot(self.H_a)).dot(self.S_a)

        # 4) Store the current prediction
        self.mu = self.mu_a[0:3]