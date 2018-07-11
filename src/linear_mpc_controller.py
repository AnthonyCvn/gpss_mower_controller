#!/usr/bin/env python
import rospy
import numpy as np
from scipy import spatial, linalg
from math import sin, cos, pi
from orunav_msgs.msg import ControllerReport
from geometry_msgs.msg import Twist

from toolbox import TicToc, wraptopi

import cvxopt


class Controller:
    """ ...

    ...

    Attributes:
        ...: ...
    """
    def __init__(self, sampling_time=0.1, horizon_length=10):
        """ ... """
        # Map the controller status to the corresponding function.
        self.ctr_states = {ControllerReport.CONTROLLER_STATUS_ACTIVE:    self.controller_status_active,
                           ControllerReport.CONTROLLER_STATUS_FAIL:      self.controller_status_fail,
                           ControllerReport.CONTROLLER_STATUS_FINALIZE:  self.controller_status_finalize,
                           ControllerReport.CONTROLLER_STATUS_TERMINATE: self.controller_status_terminate,
                           ControllerReport.CONTROLLER_STATUS_WAIT:      self.controller_status_wait,
                           }

        # Controller report
        self.pub_robot_report = rospy.Publisher('/robot1/controller/reports', ControllerReport, queue_size=1)
        self.controller_report = ControllerReport()
        self.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT

        # Publisher for plot
        self.pub_predicted_pose = rospy.Publisher('/robot1/predicted_pose', Twist, queue_size=1)
        self.predicted_pose = Twist()
        self.pub_robot_pose = rospy.Publisher('/robot1/pose_estimate', Twist, queue_size=1)
        self.robot_pose = Twist()

        # Controller parameters
        self.Ts = sampling_time
        self.NNN = horizon_length
        self.dim_x = 3
        self.dim_u = 2

        # Reference path and trajectory
        self.ref_path = None
        self.ref_trajectory = None
        self.path_length = 0

        # Current reference index and trajectory
        self.current_trajectory = None
        self.index_path = 0
        self.distance_to_path = None
        self.max_distance_to_path = 1.0
        self.index_ahead = 7

        # MPC weights
        w_x1 = 10
        w_x2 = 10
        w_x3 = 10
        w_u1 = 1
        w_u2 = 1

        self.Q = np.kron(np.eye(self.NNN), np.diag([w_x1, w_x2, w_x3]))
        self.R = np.kron(np.eye(self.NNN), np.diag([w_u1, w_u2]))

        # MPC constraints
        self.u_max = np.array([0.7, pi])
        self.u_min = - self.u_max
        self.a_tan_max = pi/12
        self.a_tan_min = - self.a_tan_max

        # MPC variables
        self.u = np.zeros((self.dim_u, 1))
        self.u_warm_start = np.zeros((self.dim_u, 1))
        self.mu = np.zeros((self.dim_x, 1))
        self.delta_x0 = np.zeros((self.dim_x, 1))
        self.delta_x = np.zeros((self.dim_x*self.NNN, 1))

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

        # Store solver's latency
        self.latency = 0.0
        self.max_latency = 0.0
        self.first_solve_latency = 0.0

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

    def reset(self):
        self.path_length = 0
        self.index_path = 0
        self.latency = 0.0
        self.max_latency = 0.0
        self.first_solve_latency = 0.0

    def ss_model_a(self, xr, ur):
        """ ... """
        A = np.eye(self.dim_x)

        A[0, 2] = - ur[0] * self.Ts * sin(xr[2])
        A[1, 2] = + ur[0] * self.Ts * cos(xr[2])

        return A

    def ss_model_b(self, xr):
        """ ... """
        B = np.zeros((self.dim_x, self.dim_u))

        B[0, 0] = self.Ts * cos(xr[2])
        B[1, 0] = self.Ts * sin(xr[2])
        B[2, 1] = self.Ts

        return B

    def problem_construction(self):
        """ ... """
        # Compute and store all Ai and Bi  in a list (i.e. Ai = A[i])
        self.A = []
        self.B = []
        for i in range(self.NNN):
            xr = self.current_trajectory[i, 0:self.dim_x]
            ur = self.current_trajectory[i, self.dim_x:self.dim_x+self.dim_u]
            self.A.append(self.ss_model_a(xr, ur))
            self.B.append(self.ss_model_b(xr))


        # Compute matrices 'S' and 'T' that describe the dynamic system x+ = S * x0 + T * u
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
        """
        Solve a Quadratic Program defined as:

            minimize
                (1/2) * x.T * P * x + q.T * x

            subject to
                G * x <= h

        using CVXOPT <http://cvxopt.org/>.

        Parameters
        ----------
        P : numpy.array, cvxopt.matrix or cvxopt.spmatrix
            Symmetric quadratic-cost matrix.
        q : numpy.array, cvxopt.matrix or cvxopt.spmatrix
            Quadratic-cost vector.
        G : numpy.array, cvxopt.matrix or cvxopt.spmatrix
            Linear inequality matrix.
        h : numpy.array, cvxopt.matrix or cvxopt.spmatrix
            Linear inequality vector.

        initvals : numpy.array, optional
            Warm-start guess vector.

        Returns
        -------
        x : array
            Solution to the QP, if found, otherwise ``None``.

        Note
        ----
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
            rospy.loginfo("The solver did not find an optimal solution.")
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_FAIL
            self.ctr_states[self.controller_report.status]()

        return sol

    def compute(self):
        """ ... """
        # Call states' functions
        self.ctr_states[self.controller_report.status]()

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

    def trajectory_selector(self):
        # Find the nearest point on the path
        # self.index_path = spatial.KDTree(self.ref_path[:, 0:2]).query(self.mu[0:2].T)[1][0]
        if self.index_path + self.index_ahead > self.path_length:
            self.index_ahead = self.path_length - self.index_path

        self.distance_to_path = linalg.norm((self.ref_path[self.index_path, 0:2] - self.mu[0:2].T))
        for i in range(self.index_path + 1, self.index_path + self.index_ahead):
            distance_to_path_i = linalg.norm((self.ref_path[i, 0:2] - self.mu[0:2].T))
            if distance_to_path_i < self.distance_to_path:
                self.distance_to_path = distance_to_path_i
                self.index_path = i

        # Select the current reference trajectory according to horizon's length
        if self.path_length - self.index_path >= self.NNN:
            self.current_trajectory = self.ref_trajectory[self.index_path:self.NNN+self.index_path, :]
        else:
            index_left = self.path_length - self.index_path
            self.current_trajectory = np.zeros([self.NNN, 5])
            self.current_trajectory[0:index_left, :] = self.ref_trajectory[self.index_path::, :]
            self.current_trajectory[index_left:self.NNN, :] = self.ref_trajectory[-1, :]

    def state_subtraction(self, x1, x2):
        delta = x1 - x2
        delta[2] = wraptopi(delta[2])
        return delta

    def controller_status_active(self):
        """ ... """
        self.trajectory_selector()

        if self.distance_to_path > self.max_distance_to_path:
            rospy.loginfo("The path is too far away from the mower.")
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_FAIL
            self.ctr_states[self.controller_report.status]()

        x0r = self.current_trajectory[0, 0:self.dim_x]

        self.delta_x0 = self.state_subtraction(self.mu, x0r[:, None])

        self.problem_construction()

        t = TicToc()
        t.tic()

        sol = self.cvxopt_solve_qp(self.P, self.q, self.G, self.g, self.u_warm_start)

        self.latency = t.toc()*1000
        if self.latency > self.max_latency:
            if self.first_solve_latency == 0:
                self.first_solve_latency = self.latency
            else:
                self.max_latency = self.latency

        print "Latency     [ms]: ", self.latency
        print "First solve [ms]: ", self.first_solve_latency
        print "Max         [ms]: ", self.max_latency
        print ""

        self.u[0] = sol['x'][0] + self.current_trajectory[0, self.dim_x]
        self.u[1] = sol['x'][1] + self.current_trajectory[0, self.dim_x + 1]

        # Publish predicted poses
        #self.delta_u = np.array(sol['x'])
        #self.delta_x = self.S.dot(self.delta_x0) + self.T.dot(self.delta_u)
        #if self.index_path % (2*self.NNN) == 0:
        #    print"path index", self.index_path
        #    for i in range(1, self.NNN):
        #        self.predicted_pose.linear.x = self.delta_x[(i-1)*self.dim_x] + self.current_trajectory[i, 0]
        #        self.predicted_pose.linear.y = self.delta_x[(i-1)*self.dim_x+1] + self.current_trajectory[i, 1]
        #        self.predicted_pose.angular.z = self.delta_x[(i-1)*self.dim_x+2] + self.current_trajectory[i, 2]
        #        self.pub_predicted_pose.publish(self.predicted_pose)
        #        print"Predicted path, x "+str(i)+" = ", self.predicted_pose.linear.x
        #        print"Predicted path, y " + str(i) + " = ", self.predicted_pose.linear.y

        #    print"Ref path: ", self.current_trajectory
        #    print""

        self.u_warm_start = self.u

        if self.index_path == self.path_length - 1:
            rospy.loginfo("Robot{0} wait for a new path.".format(self.controller_report.robot_id))
            self.reset()
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            self.ctr_states[self.controller_report.status]()

    def controller_status_fail(self):
        """ ... """
        self.u[0] = 0.0
        self.u[1] = 0.0

    def controller_status_finalize(self):
        """ ... """
        self.u[0] = 0.0
        self.u[1] = 0.0

    def controller_status_terminate(self):
        """ ... """
        self.u[0] = 0.0
        self.u[1] = 0.0

    def controller_status_wait(self):
        """ ... """
        self.u[0] = 0.0
        self.u[1] = 0.0
