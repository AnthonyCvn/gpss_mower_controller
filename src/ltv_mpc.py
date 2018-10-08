#!/usr/bin/env python

# ROS libraries for Python.
import rospy

# ROS messages.
from orunav_msgs.msg import ControllerReport
from geometry_msgs.msg import Twist

# Python packages.
import os
import numpy as np
from scipy import linalg
from math import sin, cos
import cvxopt

# Specific controller's libraries.
from toolbox import TicToc


class Regulator:
    """ Trajectory following Controller based on Model Predictive Control (MPC).

    The Controller can be in the following states:
        - Active
          Compute the input command of the robot:
            1) Trajectory selector.
            2) Construct the Quadratic Problem (QP) according to the MPC formulation.
            3) Solve the QP
            4) Store the result command input as a warm start for the next QP.

        - Fail
          Stop the robot in case of a fault. For example when is too far compare to the path.

        - Finalize
          The robot reached the destination

        - Terminate
          Stop the robot smoothly. For example when a new trajectory is set.

        - Wait
          Wait for a new trajectory to follow.

    Note:
        A report about the controller's state is published at sampling frequency.

    """
    def __init__(self, robot_id=1, sampling_time=0.1, horizon_length=10, weights=None, constraints=None):

        # Map the controller status to the corresponding function.
        self.ctr_states = {ControllerReport.CONTROLLER_STATUS_ACTIVE:    self.controller_status_active,
                           ControllerReport.CONTROLLER_STATUS_FAIL:      self.controller_status_fail,
                           ControllerReport.CONTROLLER_STATUS_FINALIZE:  self.controller_status_finalize,
                           ControllerReport.CONTROLLER_STATUS_TERMINATE: self.controller_status_terminate,
                           ControllerReport.CONTROLLER_STATUS_WAIT:      self.controller_status_wait,
                           }

        # Controller status flag and check points
        self.controller_active = False
        self.check_points = [0]

        # Controller report
        self.pub_robot_report = rospy.Publisher('/robot{0}/controller/reports'.format(robot_id),
                                                ControllerReport, queue_size=1)
        self.controller_report = ControllerReport()
        self.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
        self.controller_report.robot_id = robot_id

        # Publisher for plot
        self.pub_robot_pose = rospy.Publisher('/robot{0}/mu'.format(robot_id), Twist, queue_size=1)
        self.robot_pose = Twist()

        # Controller parameters
        self.Ts = sampling_time
        self.NNN = horizon_length
        self.dim_x = 3
        self.dim_u = 2

        # Reference path and trajectory
        self.sub_trajectories = None
        self.n_subgoal = 0
        self.ref_path = None
        self.ref_trajectory = None
        self.path_length = 0

        # Current reference index and trajectory
        self.current_trajectory = None
        self.index_path = 0
        self.distance_to_path = None
        self.max_distance_to_path = 1.0
        self.index_ahead = 10
        self.index_back = 10
        self.final_index_counter = 0

        # Read MPC weights and constraints. Set defaults value if not set.
        try:
            rospy.loginfo("MPC weights: {0}, {1}, {2}, {3}, {4} "
                          .format(weights[0], weights[1], weights[2], weights[3], weights[4]))
        except IndexError:
            weights = [10, 10, 10, 1, 1]
            rospy.loginfo("MPC weights set to default: {0}, {1}, {2}, {3}, {4} "
                          .format(weights[0], weights[1], weights[2], weights[3], weights[4]))
        try:
            rospy.loginfo("MPC constraints: {0}, {1}, {2}, {3}, {4} "
                          .format(constraints[0], constraints[1], constraints[2],
                                  constraints[3], constraints[4], constraints[5]))
        except IndexError:
            constraints = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5]
            rospy.loginfo("MPC constraints set to default: {0}, {1}, {2}, {3}, {4}"
                          .format(constraints[0], constraints[1], constraints[2],
                                  constraints[3], constraints[4], constraints[5]))

        # MPC weights
        diagonal = []
        for i in range(self.NNN-1):
            diagonal.append(2**i)
        diagonal.append(2**i*30)

        self.Q = np.kron(np.diag(diagonal), np.diag([weights[0], weights[1], weights[2]]))

        #self.Q = np.kron(np.eye(self.NNN), np.diag([weights[0], weights[1], weights[2]]))
        self.R = np.kron(np.eye(self.NNN), np.diag([weights[3], weights[4]]))

        # MPC constraints
        self.u_max = np.array([constraints[0], constraints[2]])
        self.u_min = np.array([constraints[1], constraints[3]])
        self.a_tan_max = constraints[4]
        self.a_tan_min = constraints[5]


        # MPC variables
        self.u = np.zeros((self.dim_u, 1))
        self.u_warm_start = np.zeros((self.dim_u, 1))
        self.mu = np.zeros((self.dim_x, 1))
        self.delta_x0 = np.zeros((self.dim_x, 1))
        self.delta_x = np.zeros((self.dim_x*self.NNN, 1))
        self.delta_u = np.zeros((self.dim_u * self.NNN, 1))

        self.A = []
        self.B = []
        self.S = None
        self.T = None
        self.P = None
        self.q = None
        self.D = np.vstack((np.eye(self.dim_u), -np.eye(self.dim_u)))

        # Construct constraints matrices such that G * u <= g
        H_line = np.zeros(2 * self.NNN)
        H_line[0] = -1
        H_line[2] = 1
        self.H = np.empty([0, 2 * self.NNN])
        for i in range(self.NNN - 1):
            self.H = np.vstack((self.H, np.roll(H_line, 2 * i)))

        self.H = np.vstack((self.H, -self.H))

        self.G = np.kron(np.eye(self.NNN), self.D)
        self.G = np.vstack((self.G, self.H))
        self.g = None

        # Store solver's latency and print result
        self.solver_latency = 0.0
        self.max_latency = 0.0
        self.first_solve_latency = 0.0
        self.print_head_file = True

        # Solver options
        # (default show_progress = True)
        cvxopt.solvers.options['show_progress'] = False
        # (default maxiters = 100)
        cvxopt.solvers.options['maxiters'] = 100
        # (default abstol = 1e-7)
        cvxopt.solvers.options['abstol'] = 1e-7
        # (default reltol = 1e-6)
        cvxopt.solvers.options['reltol'] = 1e-6
        # (default feastol = 1e-7)
        cvxopt.solvers.options['feastol'] = 1e-7

    def compute(self, pose):
        """ Function called at sampling frequency to compute the input command.

            Switch between call-states functions to compute the input command.

        """
        self.mu = pose

        # Call states' functions
        self.ctr_states[self.controller_report.status]()

        return self.u

    def controller_status_active(self):
        """ When the controller is in active mode; comute the command input to follow the reference trajectory.

        1) Trajectory selector.
        2) Construct the Quadratic Problem (QP) according to the MPC formulation.
        3) Solve the QP
        4) Log warm start vector for the next QP.
        5) Check logic to exit the active state.

        """
        self.controller_active = True

        # 1) Trajectory selector.
        self.trajectory_selector()

        if self.distance_to_path > self.max_distance_to_path:
            rospy.loginfo("Robot #{0} is too far from the path."
                          .format(self.controller_report.robot_id))
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_FAIL
            self.ctr_states[self.controller_report.status]()

        # 2) Construct the Quadratic Problem (QP) according to the MPC formulation.
        x0r = self.current_trajectory[0, 0:self.dim_x]
        self.delta_x0 = self.state_substraction(self.mu, x0r[:, None])
        self.problem_construction()

        # 3) Solve the QP
        t = TicToc()
        t.tic()
        sol = self.cvxopt_solve_qp(self.P, self.q, self.G, self.g, self.u_warm_start)
        self.solver_latency = t.toc()

        self.delta_u = np.array(sol['x'])

        self.u[0] = sol['x'][0] + self.current_trajectory[0, self.dim_x]
        self.u[1] = sol['x'][1] + self.current_trajectory[0, self.dim_x + 1]

        # 4) Log warm start vector for the next QP.
        self.u_warm_start = self.u

        # 5) Logic to exit the active state.
        segment_speed = self.current_trajectory[0, self.dim_x]
        if segment_speed == 0.0:
            if self.n_subgoal > 1:
                self.update_trajectory()
            else:
                self.final_index_counter += 1
                if self.final_index_counter > 2:
                    self.final_index_counter = 0
                    self.controller_report.status = ControllerReport.CONTROLLER_STATUS_FINALIZE

    def controller_status_fail(self):
        """ Fail Status; Stop the robot because of a fault """
        self.controller_active = False
        self.u[0] = 0.0
        self.u[1] = 0.0

    def controller_status_finalize(self):
        """ Finalize Status; The robot reached the destination """
        self.controller_active = False
        self.u[0] = 0.0
        self.u[1] = 0.0
        rospy.loginfo("Robot #{0} is ready to receive a new task.".format(self.controller_report.robot_id))
        self.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT

    def controller_status_terminate(self):
        """ Stop the robot smoothly """
        self.controller_active = False
        if self.u[0] == 0.0 and self.u[1] == 0.0:
            self.reset()
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            self.ctr_states[self.controller_report.status]()

        if self.u[0] > 0.0 or self.u[1] > 0.0:
            self.u[0] -= 0.1
            self.u[1] -= 0.1
            if self.u[0] < 0.1:
                self.u[0] = 0.0
            if self.u[1] < 0.1:
                self.u[1] = 0.0

        if self.u[0] < 0.0 or self.u[1] < 0.0:
            self.u[0] += 0.1
            self.u[1] += 0.1
            if self.u[0] > -0.1:
                self.u[0] = 0.0
            if self.u[1] > -0.1:
                self.u[1] = 0.0

    def controller_status_wait(self):
        """ Wait Status; Wait for a new trajectory to follow. """
        self.controller_active = False
        self.u[0] = 0.0
        self.u[1] = 0.0

    def ss_model_a(self, xr, ur):
        """ Computation of matrix A(k) that is part of the state space model x(k+1) = A(k) * x(k) + B(k) """
        A = np.eye(self.dim_x)

        A[0, 2] = - ur[0] * self.Ts * sin(xr[2])
        A[1, 2] = + ur[0] * self.Ts * cos(xr[2])

        return A

    def ss_model_b(self, xr):
        """ Computation of matrix B(k) that is part of the state space model x(k+1) = A(k) * x(k) + B(k) """
        B = np.zeros((self.dim_x, self.dim_u))

        B[0, 0] = self.Ts * cos(xr[2])
        B[1, 0] = self.Ts * sin(xr[2])
        B[2, 1] = self.Ts

        return B

    def problem_construction(self):
        """ Construct the Quadratic Problem (QP) according to the MPC formulation. """
        # Compute and store all Ai and Bi  in a list (i.e. Ai = A[i])
        self.A = []
        self.B = []
        for i in range(self.NNN):
            xr = self.current_trajectory[i, 0:self.dim_x]
            ur = self.current_trajectory[i, self.dim_x:self.dim_x+self.dim_u]
            self.A.append(self.ss_model_a(xr, ur))
            self.B.append(self.ss_model_b(xr))

        # Compute matrices 'S' and 'T' that describe the dynamic system x(k+1) = S(k) * x(0) + T * u(k)
        self.S = self.A[0]
        S_work = self.A[0]
        for i in range(1, self.NNN):
            S_work = self.A[i].dot(S_work)
            self.S = np.vstack((self.S, S_work))

        self.T = np.empty([self.dim_x * self.NNN, 0])
        for j in range(self.NNN):
            T_colonne = self.B[j]
            T_colonne_work = self.B[j]
            for i in range(j+1, self.NNN):
                T_colonne_work = self.A[i].dot(T_colonne_work)
                T_colonne = np.vstack((T_colonne, T_colonne_work))

            T_colonne = np.vstack((np.zeros([j*self.dim_x, self.dim_u]), T_colonne))
            self.T = np.hstack((self.T, T_colonne))

        # Compute the symmetric quadratic-cost matrix P and the cost vector q
        self.P = self.T.T.dot(self.Q).dot(self.T) + self.R
        self.q = 2 * self.T.T.dot(self.Q).dot(self.S).dot(self.delta_x0)

        # Compute inequality constraints matrices
        # First bounds on the control variables u_min <= u <= u_max
        # Then bounds the tangential acceleration with L_min <= H * u <= L_max
        # Finally, add all the constraints on the system G * u <= g
        self.g = np.empty([0, 1])

        for i in range(self.NNN):
            delta_u_max = self.u_max - self.current_trajectory[i, self.dim_x:self.dim_x+self.dim_u]
            delta_u_min = self.u_min - self.current_trajectory[i, self.dim_x:self.dim_x+self.dim_u]
            # Numpy doesn't return the transpose of a 1D array; instead we use: vector[:, None]
            self.g = np.vstack((self.g, np.vstack((delta_u_max[:, None], -delta_u_min[:, None]))))

        for i in range(1, self.NNN):
            v_i_m = self.current_trajectory[i - 1, self.dim_x]
            v_i = self.current_trajectory[i, self.dim_x]
            self.g = np.vstack((self.g, self.Ts*self.a_tan_max - v_i + v_i_m))

        for i in range(1, self.NNN):
            v_i_m = self.current_trajectory[i - 1, self.dim_x]
            v_i = self.current_trajectory[i, self.dim_x]
            self.g = np.vstack((self.g, - self.Ts * self.a_tan_min + v_i - v_i_m))

    def cvxopt_solve_qp(self, P_np, q_np, G_np, g_np, x_init_np):
        """ Solve a Quadratic Program defined by numpy (np) arrays as argument.

        Solve a Quadratic Program defined as:

            minimize
                (1/2) * x.T * P * x + q.T * x

            subject to
                G * x <= g

        using CVXOPT <http://cvxopt.org/>.

        Args:
            P_np        : Symmetric quadratic-cost matrix.
            q_np        : Quadratic-cost vector.
            G_np        : Linear inequality matrix.
            g_np        : Linear inequality vector.
            x_init_np   : numpy.array, Warm-start vector.

        Returns:
            x : Solution to the QP.

        Note:
            CVXOPT only considers the lower entries of `P`, therefore it will use a
            wrong cost function if a non-symmetric matrix is provided.
        """
        # Convert numpy to cvxopt matrix with tc='d' option to explicitly construct a matrix of doubles
        P = cvxopt.matrix(P_np, tc='d')
        q = cvxopt.matrix(q_np, tc='d')
        G = cvxopt.matrix(G_np, tc='d')
        g = cvxopt.matrix(g_np, tc='d')
        x_init = cvxopt.matrix(x_init_np, tc='d')

        sol = cvxopt.solvers.qp(P, q, G, g, initvals=x_init)

        if sol['status'] == 'unknown':
            rospy.loginfo("Robot #{0} failed to find an optimal solution."
                          .format(self.controller_report.robot_id))
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_FAIL
            self.ctr_states[self.controller_report.status]()

        return sol

    def trajectory_selector(self):
        """ Select the trajectory that the robot should follow. """
        # Find the nearest point on the path
        if self.index_path < self.index_back:
            self.index_back = self.index_path

        if self.index_path + self.index_ahead > self.path_length:
            self.index_ahead = self.path_length - self.index_path

        self.distance_to_path = linalg.norm((self.ref_path[self.index_path - self.index_back, 0:2] - self.mu[0:2].T))
        for i in range(self.index_path - self.index_back + 1, self.index_path + self.index_ahead):
            distance_to_path_i = linalg.norm((self.ref_path[i, 0:2] - self.mu[0:2].T))
            if distance_to_path_i < self.distance_to_path:
                self.distance_to_path = distance_to_path_i
                self.index_path = i + 1

        # Select the current reference trajectory according to the horizon and index_path
        if self.path_length - self.index_path > self.NNN:
            self.current_trajectory = self.ref_trajectory[self.index_path:self.index_path+self.NNN + 1, :]
        elif self.path_length > 0:
            index_left = self.path_length - self.index_path
            self.current_trajectory = np.zeros([self.NNN + 1, 5])
            self.current_trajectory[0:index_left, :] = self.ref_trajectory[self.index_path::, :]
            self.current_trajectory[index_left:self.NNN + 1, :] = self.ref_trajectory[-1, :]

    def update_trajectory(self):
        """ Update the trajectory and parameters."""
        self.n_subgoal = len(self.sub_trajectories)
        self.ref_trajectory = self.sub_trajectories.pop(0)
        self.ref_path = self.ref_trajectory[:, 0:3]
        self.path_length = len(self.ref_path)

        # Find the current index (closest to the robot)
        self.index_path = 0
        self.distance_to_path = linalg.norm((self.ref_path[0, 0:2] - self.mu[0:2].T))
        for i in range(1, len(self.ref_path)):
            distance_to_path_i = linalg.norm((self.ref_path[i, 0:2] - self.mu[0:2].T))
            if distance_to_path_i < self.distance_to_path:
                self.distance_to_path = distance_to_path_i
                self.index_path = i

    def reset(self):
        """ Reset variables to zero. """
        self.sub_trajectories = []
        self.path_length = 0
        self.index_path = 0
        self.solver_latency = 0.0
        self.max_latency = 0.0
        self.first_solve_latency = 0.0

    @staticmethod
    def wraptopi(angle):
        """ Wrap angle between -pi and pi. """
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    def state_substraction(self, x1, x2):
        """ Substraction of two robot's state x1 and x2.
        Args:
            x1      : First pose (x1, y1, phi1).
            x2      : Second pose (x2, y2, phi2).

        Returns:
            delta   : (x, y, phi) = x1 - x2 with the angles phi wrapped between -pi and pi.
        """
        delta = x1 - x2
        delta[2] = self.wraptopi(delta[2])
        return delta

    def publish_report(self):
        """ Publish the controller's report. """
        # Publish controller report
        self.controller_report.state.position_x = self.mu[0]
        self.controller_report.state.position_y = self.mu[1]
        self.controller_report.state.orientation_angle = self.mu[2]
        self.controller_report.stamp = rospy.get_rostime()

        self.pub_robot_report.publish(self.controller_report)

        # Publish graph's data
        self.robot_pose.linear.x = self.mu[0]
        self.robot_pose.linear.y = self.mu[1]
        self.robot_pose.angular.z = self.mu[2]
        self.pub_robot_pose.publish(self.robot_pose)

    def log_to_file(self, filename, photo):
        # calculate the predicted poses
        self.delta_x = self.S.dot(self.delta_x0) + self.T.dot(self.delta_u)

        with open(filename, "a+") as my_file:
            if self.print_head_file:
                self.print_head_file = False
                my_file.write("x_ref,y_ref,phi_ref,x_pred,y_pred,phi_pred,v_ref,w_ref,v_pred,w_pred,photo_x,photo_y,latency\n")

            x0_ref = self.current_trajectory[0, 0]
            y0_ref = self.current_trajectory[0, 1]
            phi0_ref = self.current_trajectory[0, 2]

            x0 = self.mu[0]
            y0 = self.mu[1]
            phi0 = self.mu[2]

            v0_ref = self.current_trajectory[0, self.dim_x]
            w0_ref = self.current_trajectory[0, self.dim_x + 1]

            v0 = self.delta_u[0] + v0_ref
            w0 = self.delta_u[0] + w0_ref

            photo_x = photo[0]
            photo_y = photo[1]

            my_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n"
                          .format(x0_ref, y0_ref, phi0_ref, x0[0], y0[0], phi0[0],
                                  v0_ref, w0_ref, v0[0], w0[0], photo_x[0], photo_y[0], self.solver_latency))

            for i in range(1, self.NNN):
                x_ref = self.current_trajectory[i, 0]
                y_ref = self.current_trajectory[i, 1]
                phi_ref = self.current_trajectory[i, 2]

                x_pred = self.delta_x[(i-1)*self.dim_x] + x_ref
                y_pred = self.delta_x[(i-1)*self.dim_x+1] + y_ref
                phi_pred = self.delta_x[(i-1)*self.dim_x+2] + phi_ref

                v_ref = self.current_trajectory[i, self.dim_x]
                w_ref = self.current_trajectory[i, self.dim_x + 1]

                v_pred = self.delta_u[i*self.dim_u] + v_ref
                w_pred = self.delta_u[i*self.dim_u + 1] + w_ref

                my_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},,,\n"
                              .format(x_ref, y_ref, phi_ref, x_pred[0], y_pred[0],
                                      phi_pred[0], v_ref, w_ref, v_pred[0], w_pred[0]))

            x_ref = self.current_trajectory[self.NNN, 0]
            y_ref = self.current_trajectory[self.NNN, 1]
            phi_ref = self.current_trajectory[self.NNN, 2]

            x_pred = self.delta_x[(self.NNN-1) * self.dim_x] + x_ref
            y_pred = self.delta_x[(self.NNN-1) * self.dim_x + 1] + y_ref
            phi_pred = self.delta_x[(self.NNN-1) * self.dim_x + 2] + phi_ref

            my_file.write("{0},{1},{2},{3},{4},{5},,,,,,,\n"
                          .format(x_ref, y_ref, phi_ref, x_pred[0], y_pred[0], phi_pred[0]))
