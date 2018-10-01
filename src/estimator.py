#!/usr/bin/env python

# ROS libraries for Python.
import rospy

# ROS messages.
from geometry_msgs.msg import Twist

# Python packages.
import numpy as np
from scipy import linalg
from math import sin, cos, fabs


class Estimator:
    """ Estimate the robot's pose at a given sampling frequency.

    The sensor fusion is performed by an Extended Kalman Filter (EKF).

    Args:
        robot_id            : Robot identification number.
        regulator           : Regulator that calculate the input command for the plant.
        tf_manager          : Manager that receive the sensors' data and send the TF transform between /map and /odom.

    Attributes:
        robot_id            : Store robot_id argument.
        ctrl                : Store regulator argument.
        tf_mng              : Store tf_manager argument.
        print_status        : Set True to print the controller's status in the console.
        log_to_file         : Set True to log the controller's status in a file.
        Ts                  : Sampling period
        n_calibration       : Number of iteration to calibrate the filter.
        is_calibrate        : True if the filter is calibrated.
        max_delay           : Maximum acceptable delay of the sensors in sampling period.
        odom_delay          : Delay between the odom timestamp and the begining of the sampling period in seconds.
        photo_delay         : Delay between the photo timestamp and the begining of the sampling period in seconds.
        photo_seq           : Sequence number of the photo data.
        cmd_vel             : Twist message of the velocity command.
        pub_cmd             : ROS publisher of cmd_vel.
        u                   : Input command vector.
        y                   : Sensors vector.
        mu                  : Estimate of the pose of the robot.
        G                   : Jacobian g'(x) = dg(x)/dx of the motion function; x = g(u, x) .
        S                   : Measurement covariance matrix.
        H                   : Jacobian h'(x) = dh(x)/dx of the output function; y = h(x).
        Q                   : Measurement covariance matrix.
        R                   : State transition covariance matrix.
    """

    # Estimation methods
    EKF_ODOM = 1
    EKF_PHOTO_ODOM = 2
    EKF_RECALCULATION = 3
    EKF_SAMPLE_STATE_AUGMENTATION = 4
    
    # Problem dimensions
    DIM_X = 3
    DIM_U = 2
    DIM_Y = 6

    def __init__(self, robot_id, regulator, tf_manager, method, print_status=True, log_to_file=False):
        """ ... """

        self.robot_id = robot_id
        self.ctrl = regulator
        self.tf_mng = tf_manager
        self.print_status = print_status
        self.log_to_file = log_to_file

        self.Ts = regulator.Ts
        self.n_calibration = 10
        self.is_calibrate = False
        self.max_delay = 10

        self.odom_delay = 0.0
        self.photo_delay = 0.0
        self.photo_seq = 0

        self.cmd_vel = Twist()
        self.pub_cmd = rospy.Publisher("/robot{0}/cmd_vel".format(robot_id), Twist, queue_size=1)

        rospy.loginfo("Sampling frequency set at {0} Hz".format(1.0/self.Ts))

        # Methods specific variables
        if method is self.EKF_ODOM:
            # EKF estimator variable
            self.u = np.zeros((self.DIM_U, 1))
            self.y = np.zeros((self.DIM_Y, 1))
            self.mu = np.zeros((self.DIM_X, 1))

            self.G = np.eye(self.DIM_X)
            self.H = np.vstack((np.eye(self.DIM_X), np.eye(self.DIM_X)))
            self.S = 1e-3 * np.eye(self.DIM_X)

            # Covariance matrices (Q: measurement (odometry then photogrammetry), R: state transition).
            self.Q = np.diag(np.array([25.0e-6, 25.0e-6, 4e-3, 4.0e-6, 4.0e-6, 4e-4]))
            self.R = 1.0e-1 * np.eye(self.DIM_X)

            # Start the timer
            self.y[0:3] = self.tf_mng.sensors.odom_pose
            self.mu = self.y[0:3]
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_odom_based)

        if method is Estimator.EKF_PHOTO_ODOM:
            # EKF estimator variable
            self.u = np.zeros((self.DIM_U, 1))
            self.y = np.zeros((self.DIM_Y, 1))
            self.mu = np.zeros((self.DIM_X, 1))

            self.G = np.eye(self.DIM_X)
            self.H = np.vstack((np.eye(self.DIM_X), np.eye(self.DIM_X)))
            self.S = 1e-3 * np.eye(self.DIM_X)

            # Covariance matrices (Q: measurement (odometry then photogrammetry), R: state transition).
            self.Q = np.diag(np.array([25.0e-6, 25.0e-6, 4e-3, 4.0e-6, 4.0e-6, 4e-4]))
            self.R = 1.0e-6 * np.eye(self.DIM_X)

            # Start the timer
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_photo_odom)

        if method is Estimator.EKF_SAMPLE_STATE_AUGMENTATION:
            # EKF-sample-state augmentation variable
            self.mu_a_pred = np.zeros((2 * self.DIM_X, 1))
            self.mu_a = np.zeros((2 * self.DIM_X, 1))
            self.y_a = np.zeros((self.DIM_Y, 1))
            self.G_a = np.eye(2 * self.DIM_X)
            self.H_a = np.eye(2 * self.DIM_X)
            
            # Storage of [x, y, phi] from time (k-n_prediction_stored) to time k
            self.prediction_store = []
            for j in range(self.n_calibration):
                self.prediction_store.append(np.zeros((self.DIM_X, 1)))

            # Covariance matrices for augmented states
            # S_a: EKF covariance matrix.
            # Q_: measurement (odom then photo)
            # R_a: state transition (new states then delayed states)
            self.S_a = 1e-3 * np.ones((2 * self.DIM_X, 2 * self.DIM_X))
            self.Q_a = np.diag(np.array([4.0e-4, 4.0e-4, 4.0e-4, 4.0e-6, 4.0e-6, 4.0e-5]))
            self.R_a = np.diag(np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6, 1.0e-5]))

            # Start the timer
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_photo_odom_augmented)

        if method is Estimator.EKF_RECALCULATION:
            # EKF estimator variable
            self.u = np.zeros((self.DIM_U, 1))
            self.y = np.zeros((self.DIM_Y, 1))
            self.mu = np.zeros((self.DIM_X, 1))

            self.G = np.eye(self.DIM_X)
            self.H = np.vstack((np.eye(self.DIM_X), np.eye(self.DIM_X)))
            self.S = 1e-3 * np.eye(self.DIM_X)

            # Covariance matrices (Q: measurement (odom then photo), R: state transition).
            self.Q = np.diag(np.array([25.0e-6, 25.0e-6, 4e-3, 4.0e-6, 4.0e-6, 4e-4]))
            self.R = 1.0e-6 * np.eye(self.DIM_X)

            # Store [mu, sigma, u, z] from time (k-max_delay) to time k
            self.store = []
            store_row = [np.copy(self.mu), np.copy(self.S), np.copy(self.u), np.copy(self.y)]
            for j in range(self.max_delay):
                self.store.append(store_row)

            # Start the timer
            rospy.Timer(rospy.Duration(self.Ts), self.timer_cb_photo_odom_recalculation)

    def timer_cb_odom_based(self, event):
        """ Timer callback running at sampling frequency (based only on odometry).

        For every sampling period:
        1) Store variables.
        2) Estimate the pose of the robot.
        3) Compute the command inputs.
        4) Send the velocity command.
        5) Send the controller's report.
        6) Update the pose of the robot in the transform manager.
        7) Log the controller's status in a file if "log_to_file" is True.
        8) Print status in the console if "print_status" is True.

        """
        # 1) Store variables.
        t = rospy.get_rostime().now().to_sec()
        self.y[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot
        self.odom_delay = t - self.tf_mng.sensors.odom_t

        if self.odom_delay > 2*self.Ts:
            rospy.loginfo("Robot #{0} has Large sensor-to-controller delay of {1} ms !"
                          .format(self.robot_id, self.odom_delay*1000))

        # 2) Estimate the pose of the robot.
        self.ekf()

        # 3) Compute the command inputs.
        self.u = self.ctrl.compute(self.mu)

        # 4) Send the velocity command.
        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        # 5) Send the controller's report.
        self.ctrl.publish_report()

        # 6) Update the pose of the robot in the transform manager.
        self.tf_mng.update_world2odom(self.mu, T_odom2robot)

        # 7) Log the controller's status in a file if "log_to_file" is True.
        if self.log_to_file:
            if self.ctrl.controller_active:
                filename = "log_N{0}_f{1}.txt".format(self.ctrl.NNN, int(1.0 / self.Ts))
                self.ctrl.log_to_file(filename)

        # 8) Print status in the console if "print_status" is True
        if self.print_status:
            if self.ctrl.controller_active:
                index = self.ctrl.index_path
                total_index = self.ctrl.path_length
                percent = 100.0 * index / (1.0 * total_index)
                segment_speed_v = self.ctrl.current_trajectory[0, 3]
                segment_speed_w = self.ctrl.current_trajectory[0, 4]
                solver_latency = self.ctrl.solver_latency
                computation_delay = rospy.get_rostime().now().to_sec() - t
                print"\n\n\n" \
                     "Index:             {0}/{1} ({2}%)     \n" \
                     "Segment reference: {3} m/s {4} rad/ms \n" \
                     "Command:           {5} m/s {6} rad/ms \n" \
                     "Sampling time:     {7} ms             \n" \
                     "Odom delay:        {8} ms             \n" \
                     "Computation delay: {9} ms             \n" \
                     "Solver latency:    {10} ms            \n" \
                    .format(index, total_index, int(percent), round(segment_speed_v, 2), round(segment_speed_w*1000, 2),
                            round(self.u[0], 2), round(self.u[1]*1000, 2), self.Ts*1000, round(self.odom_delay*1000, 1),
                            round(computation_delay*1000, 1), round(solver_latency*1000, 1))

    def timer_cb_photo_odom_augmented(self, event):
        """ Timer callback running at sampling frequency with augmented states (based on odometry and photogramtery) """
        ### Diego ### self.y[0:3] = self.tf_mng.get_world2robot(self.tf_mng.odom_odom2robot)
        self.y[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot

        # Calibration process
        if not self.is_calibrate:
            if self.tf_mng.sensors.is_photo:
                # Init. the robot's poses (current and stored) with the tag's pose.
                self.mu = self.tf_mng.get_photo_pose(self.tf_mng.photo_world2robot)
                self.y[0:3] = self.mu
                for j in range(self.n_calibration):
                    self.prediction_store[j] = self.mu
                self.is_calibrate = True
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
            if sample_delay >= self.n_calibration:
                sample_delay = 0

            # Store and recover the past prediction
            self.prediction_store.append(self.mu_a_pred[0:3])
            self.prediction_store.pop(0)
            self.mu_a_pred[3:6] = self.prediction_store[-sample_delay]

            # Case without delay
            if sample_delay == 0:
                self.mu_a_pred[3:6] = self.motion_model(self.u, self.mu, self.Ts)

            self.ekf_augmented()
        else:
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0

            self.ekf()

        self.u = self.ctrl.compute(self.mu)

        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.ctrl.publish_report()

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)

    def timer_cb_photo_odom_recalculation(self, event):
        """ Timer callback running at sampling frequency (based on odometry and photogramtery) """
        # Storage
        store_row = [np.copy(self.mu), np.copy(self.S), np.copy(self.u), np.copy(self.y)]
        self.store.append(store_row)
        self.store.pop(0)

        T_odom2robot = self.tf_mng.T_odom2robot

        # Read sensors
        odom = np.copy(self.tf_mng.odom_pose)
        photo = np.copy(self.tf_mng.photo_pose)

        # Calibration process
        if not self.is_calibrate:
            if self.tf_mng.sensors.is_photo:
                # Init. the robot's poses (current and stored) with the tag pose.
                self.mu = np.copy(photo)
                self.y[0:3] = np.copy(photo)
                self.y[3:6] = np.copy(photo)
                store_row = [np.copy(self.mu), np.copy(self.S), np.copy(self.u), np.copy(self.y)]
                for j in range(self.max_delay):
                    self.store[j] = store_row
                self.is_calibrate = True
                self.tf_mng.update_world2odom(self.mu, T_odom2robot)
                rospy.loginfo("Robot #{0} set its pose according to hrp0{0} tag.".format(self.robot_id))
                return
            else:
                return

        self.odom_delay = event.current_real.to_sec() - self.tf_mng.sensors.odom_t

        if self.odom_delay > 10*self.Ts:
            rospy.loginfo("Robot #{0} has Large sensor-to-controller delay of {1} ms !"
                          .format(self.robot_id, self.odom_delay*1000))

        if self.tf_mng.sensors.is_photo:
            self.tf_mng.sensors.is_photo = False

            # Calculate the delay in number of samples
            self.delay_photo = event.current_real.to_sec() - self.tf_mng.photo_world2robot.header.stamp.to_sec()
            sample_delay = int(round(self.delay_photo/self.Ts))
            if sample_delay >= self.n_calibration:
                sample_delay = 0

            for s in range(sample_delay, 0, -1):
                self.u = np.copy(self.store[-(s+1)][2])
                self.y = np.copy(self.store[-s][3])
                if s == sample_delay:
                    self.mu = np.copy(self.store[-(s + 1)][0])
                    self.S = np.copy(self.store[-(s + 1)][1])
                    self.y[3:6] = np.copy(photo)
                    self.H[3, 0] = 1
                    self.H[4, 1] = 1
                    self.H[5, 2] = 1
                else:
                    self.H[3, 0] = 0
                    self.H[4, 1] = 0
                    self.H[5, 2] = 0
                self.ekf()

            self.u = np.copy(self.store[-1][2])
            self.y[0:3] = np.copy(odom)
            # Predict for k=0
            if sample_delay == 0:
                self.y[3:6] = np.copy(photo)
                self.H[3, 0] = 1
                self.H[4, 1] = 1
                self.H[5, 2] = 1
            else:
                self.H[3, 0] = 0
                self.H[4, 1] = 0
                self.H[5, 2] = 0
            self.ekf()

        else:
            if not self.is_calibrate:
                return
            self.y[0:3] = np.copy(odom)
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0
            self.ekf()

        self.u = self.ctrl.compute(self.mu)

        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        self.ctrl.publish_report()

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)

    def timer_cb_photo_odom(self, event):
        """ Timer callback running at sampling frequency (based on odometry and photogramtery) """
        self.y[0:3] = self.tf_mng.sensors.odom_pose
        T_odom2robot = self.tf_mng.T_odom2robot

        # Calibration process
        if not self.is_calibrate:
            if self.tf_mng.sensors.is_photo:
                # Init. the robot's poses (current and stored) with the tag's pose.
                self.mu = self.tf_mng.get_photo_pose(self.tf_mng.photo_world2robot)
                self.y[0:3] = self.mu
                self.is_calibrate = True
                rospy.loginfo("Robot #{0} set its pose according to hrp0{0} tag.".format(self.robot_id))
            else:
                return

        self.odom_delay = event.current_real.to_sec() - self.tf_mng.sensors.odom_t

        if self.odom_delay > 10*self.Ts:
            rospy.loginfo("Robot #{0} has Large sensor-to-controller delay of {1} ms !"
                          .format(self.robot_id, self.odom_delay*1000))

        if self.tf_mng.sensors.is_photo:
            self.tf_mng.sensors.is_photo = False
            self.H[3, 0] = 1
            self.H[4, 1] = 1
            self.H[5, 2] = 1

            self.y[3:6] = self.tf_mng.sensors.photo_pose

            if not self.is_calibrate:
                self.is_calibrate = True
                self.mu[0] = self.y[3]
                self.mu[1] = self.y[4]
                self.mu[2] = self.y[5]
                self.y[0] = self.y[3]
                self.y[1] = self.y[4]
                self.y[2] = self.y[5]
        else:
            if not self.is_calibrate:
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

        self.ctrl.publish_report()

        self.tf_mng.update_world2odom(self.mu, T_odom2robot)

    def motion_model(self, u, x, dt, epsilon=1e-6):
        g = np.zeros((3, 1))

        if fabs(u[1]) < epsilon:
            g[0] = x[0] + dt * u[0] * 1.0 * cos(x[2])
            g[1] = x[1] + dt * u[0] * 1.0 * sin(x[2])
            g[2] = self.wraptopi(x[2] + dt * u[1])
        else:
            g[0] = x[0] + u[0] * 1.0 / u[1] * (sin(x[2] + u[1] * dt) - sin(x[2]))
            g[1] = x[1] + u[0] * 1.0 / u[1] * (cos(x[2]) - cos(x[2] + u[1] * dt))
            g[2] = self.wraptopi(x[2] + dt * u[1])
        return g

    @staticmethod
    def jacobian_motion_model(u, x, dt, epsilon=1e-6):
        G = np.eye(3)
        if fabs(u[1]) < epsilon:
            G[0, 2] = - dt * u[0] * 1.0 * sin(x[2])
            G[1, 2] = + dt * u[0] * 1.0 * cos(x[2])
        else:
            G[0, 2] = u[0] * 1.0 / u[1] * (cos(x[2] + u[1] * dt) - cos(x[2]))
            G[1, 2] = u[0] * 1.0 / u[1] * (sin(x[2] + u[1] * dt) - sin(x[2]))

        return G

    @staticmethod
    def wraptopi(angle):
        """ Wrap angle between -pi and pi. """
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    def measure_subtraction(self, z1, z2):
        """ Substraction of two measurement z1 and z2.
        Args:
            z1      : First measurement vector (x_odom, y_odom, phi_odom, x_photo, y_photo, phi_photo).
            z2      : Second Measurement vector (x_odom, y_odom, phi_odom, x_photo, y_photo, phi_photo).

        Returns:
            delta   : (x, y, phi) =  z1 - z2 with the angles phi wrapped between -pi and pi.
        """
        delta = z1 - z2
        delta[2] = self.wraptopi(delta[2])
        delta[5] = self.wraptopi(delta[5])

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
        self.S = (np.eye(self.DIM_X) - optimal_gain.dot(self.H)).dot(self.S)

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
        self.S_a = (np.eye(2 * self.DIM_X) - optimal_gain.dot(self.H_a)).dot(self.S_a)

        # 4) Store the current prediction
        self.mu = self.mu_a[0:3]
