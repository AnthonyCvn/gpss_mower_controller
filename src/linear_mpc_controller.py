#!/usr/bin/env python
import rospy
import numpy as np
from scipy import spatial, linalg
from math import sin, cos
from orunav_msgs.msg import ControllerReport

from toolbox import TicToc

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

        # Controller parameters
        self.Ts = sampling_time
        self.NNN = horizon_length
        self.dim_x = 3
        self.dim_u = 2
        self.max_distance_to_path = 0.3

        # Reference path and trajectory
        self.ref_path = None
        self.ref_trajectory = None
        self.path_length = 0

        # Current reference index and trajectory
        self.current_trajectory = None
        self.index_path = None
        self.distance_to_path = None

        # MPC weights
        w_x1 = 1
        w_x2 = 1
        w_x3 = 1
        w_u1 = 1
        w_u2 = 1

        self.Q = np.kron(np.eye(self.NNN), np.diag([w_x1, w_x2, w_x3]))
        self.R = np.kron(np.eye(self.NNN), np.diag([w_u1, w_u2]))

        # MPC constraints
        self.u_max = + 1
        self.u_min = - 1

        # MPC variables
        self.u = np.zeros((self.dim_u, 1))
        self.mu = np.zeros((self.dim_x, 1))
        self.delta_x0 = np.zeros((self.dim_x, 1))

        self.A = []
        self.B = []
        self.S = None
        self.T = None
        self.P = None
        self.q = None
        self.D = np.vstack((np.eye(self.dim_u), -np.eye(self.dim_u)))
        self.G = np.kron(np.eye(self.NNN), self.D)
        self.g = None

    def ss_model_a(self, xr, ur):
        """ ... """
        A = np.eye(3)

        A[0, 2] = - ur[0] * self.Ts * sin(xr[2])
        A[1, 2] = + ur[0] * self.Ts * cos(xr[2])

        return A

    def ss_model_b(self, xr):
        """ ... """
        B = np.zeros((3, 2))

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

        # Compute S
        self.S = self.A[0]
        S_work = self.A[0]
        for i in range(1, self.NNN):
            S_work = self.A[i].dot(S_work)
            self.S = np.vstack((self.S, S_work))

        # Compute T
        self.T = np.empty([self.dim_x * self.NNN, 0])
        for j in range(self.NNN):
            T_colonne = self.B[j]
            T_colonne_work = self.B[j]
            for i in range(j+1, self.NNN):
                T_colonne_work = self.A[i].dot(T_colonne_work)
                T_colonne = np.vstack((T_colonne, T_colonne_work))

            T_colonne = np.vstack((np.zeros([j*self.dim_x, self.dim_u]), T_colonne))
            self.T = np.hstack((self.T, T_colonne))

        # Compute P and q
        self.P = self.T.T.dot(self.Q).dot(self.T) + self.R
        self.q = 2*self.T.T.dot(self.Q).dot(self.S).dot(self.delta_x0)

        # Compute inequality constraints matrices
        self.g = np.empty([0, 1])
        for i in range(self.NNN):
            delta_u_max = self.u_max - self.current_trajectory[i, self.dim_x:self.dim_x+self.dim_u]
            delta_u_min = self.u_min - self.current_trajectory[i, self.dim_x:self.dim_x+self.dim_u]
            # Numpy doesn't return the transpose of a 1D array; instead we use: vector[:, None]
            self.g = np.vstack((self.g, np.vstack((delta_u_max[:, None], -delta_u_min[:, None]))))

    def solve_qp(self, P_np, q_np, G_np, g_np):
        # Convert numpy to cvxopt matrix with tc='d' option to explicitly construct a matrix of doubles
        P = cvxopt.matrix(P_np, tc='d')
        q = cvxopt.matrix(q_np, tc='d')
        G = cvxopt.matrix(G_np, tc='d')
        g = cvxopt.matrix(g_np, tc='d')

        return cvxopt.solvers.qp(P, q, G, g)

    def compute(self):
        """ ... """
        # Call states' function
        self.ctr_states[self.controller_report.status]()

        # Publish controller report
        self.controller_report.state.position_x = self.mu[0]
        self.controller_report.state.position_y = self.mu[1]
        self.controller_report.state.orientation_angle = self.mu[2]
        self.controller_report.stamp = rospy.get_rostime()
        self.pub_robot_report.publish(self.controller_report)

    def trajectory_selector(self):
        # Find the nearest point on the path
        self.index_path = spatial.KDTree(self.ref_path[:, 0:2]).query(self.mu[0:2].T)[1][0]
        self.distance_to_path = linalg.norm(np.array(self.ref_path[self.index_path, :]) - self.mu.T)

        # Select the current reference trajectory with horizon's length
        if self.path_length - self.index_path >= self.NNN:
            self.current_trajectory = self.ref_trajectory[self.index_path:self.NNN+self.index_path, :]
        else:
            index_left = self.path_length - self.index_path
            self.current_trajectory = np.zeros([self.NNN, 5])
            self.current_trajectory[0:index_left, :] = self.ref_trajectory[self.index_path::, :]
            self.current_trajectory[index_left:self.NNN, :] = self.ref_trajectory[-1, :]

    def controller_status_active(self):
        """ ... """
        self.trajectory_selector()

        if self.distance_to_path > self.max_distance_to_path:
            rospy.loginfo("The path is too far away from the mower.")
            self.controller_report.status = ControllerReport.CONTROLLER_STATUS_FAIL
            self.ctr_states[self.controller_report.status]()

        x0 = self.current_trajectory[0, 0:self.dim_x]
        self.delta_x0 = self.mu - x0[:, None]
        self.problem_construction()

        t = TicToc()
        t.tic()
        sol = self.solve_qp(self.P, self.q, self.G, self.g)
        print "Solver's latency [ms]: ", t.toc()*1000
        print ""
        self.u[0] = sol['x'][0] + self.ref_trajectory[self.index_path, self.dim_x]
        self.u[1] = sol['x'][1] + self.ref_trajectory[self.index_path, self.dim_x + 1]

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
