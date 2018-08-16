#!/usr/bin/env python

# ROS libraries for Python.
import rospy
import tf

# ROS messages.
from orunav_msgs.srv import ExecuteTask
from orunav_msgs.msg import ControllerReport

# Python packages.
import numpy as np
from math import sqrt, atan2, pi
from scipy import interpolate, integrate
import matplotlib.pyplot as plt

# Specific controller's libraries.
from linear_mpc_controller import Controller
from toolbox import wraptopi

plotting = False


class TaskManager:
    """ Handle task execution of the robot.

    Receive a task and transmit the information to the controller:
        1) Extract the path from ExecuteTask message.
        2) Plan a trajectory for the mobile robot.
        3) Send the trajectory to the controller.
        4) Change the controller's status to active mode.

    Attributes:
        srv_execute_task    : ROS service to execute a task given by the motion planner.
        task_srv_name       : Name of the task execution service.
        robot_id            : Robot identification number.
        controller          : Controller object.
        Ts                  : Sampling time of the controller.
    """
    def __init__(self):
        self.srv_execute_task = None
        self.task_srv_name = 'robot1/execute_task2'
        self.robot_id = 1
        self.controller = Controller()
        self.Ts = 0.1

    def run(self):
        """ Run the task manager by initializing a ROS service server. """
        rospy.loginfo("Robot{0} is ready to receive a new task.".format(self.robot_id))
        self.srv_execute_task = rospy.Service(self.task_srv_name, ExecuteTask, self.handle_task_execution)

    def handle_task_execution(self, task):
        """ Handle a new task for the controller.

        1) Extract the path from ExecuteTask message.
        2) Plan a trajectory for the mobile robot.
        3) Send the trajectory to the controller.
        4) Change the controller's status to active mode.

        """
        if len(task.task.path.path) <= 3:
            # Path way too short.
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            return 0

        self.controller.controller_report.robot_id = self.robot_id

        self.controller.sub_trajectories = self.trajectory_planning(task.task.path.path)
        self.controller.update_trajectory()

        if task.task.abort or self.controller.path_length == 0:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
        else:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE

        return 0

    def trajectory_planning(self, path):
        """ Trajectory planner that assume a constant velocity between each points on the path.

        Args:
            path            : List of the robot's poses (x, y, phi) along the path.

        Returns:
            ref_trajectory  : Trajectory (x, y, phi, v, w) along the path (x, y, phi)

        """
        # Split the path with a different goal point at each change of direction
        goal_index = []
        goal_index.append(0)
        for i in range(len(path) - 2):
            yaw0 = atan2(path[i+1].pose.position.y - path[i].pose.position.y,
                         path[i+1].pose.position.x - path[i].pose.position.x)
            yaw1 = atan2(path[i+2].pose.position.y - path[i+1].pose.position.y,
                         path[i+2].pose.position.x - path[i+1].pose.position.x)
            if abs(wraptopi(yaw1 - yaw0)) > pi / 2:
                goal_index.append(i + 1)
        goal_index.append(len(path) - 1)
        N_subpath = len(goal_index)-1

        # Separate the path for each goal
        sub_path = []
        for i in range(N_subpath):
            sub_path.append(path[goal_index[i]:goal_index[i+1]])

        # Check for each subpath if it is reversed
        is_subpath_reversed = []
        for s in range(N_subpath):
            yaw0 = atan2(sub_path[s][1].pose.position.y - sub_path[s][0].pose.position.y,
                         sub_path[s][1].pose.position.x - sub_path[s][0].pose.position.x)

            quaternion_subpath0 = (sub_path[s][1].pose.orientation.x,
                               sub_path[s][0].pose.orientation.y,
                               sub_path[s][0].pose.orientation.z,
                               sub_path[s][0].pose.orientation.w)
            euler_subpath0 = tf.transformations.euler_from_quaternion(quaternion_subpath0, axes='sxyz')
            phi0 = euler_subpath0[2]

            if abs(wraptopi(phi0 - yaw0)) > pi / 2:
                is_subpath_reversed.append(True)
            else:
                is_subpath_reversed.append(False)

        # Create the references trajectories
        ref_trajectory = []
        bspline = []
        segment_length = 0.05
        for s in range(N_subpath):
            x = np.array([])
            y = np.array([])
            for i in range(len(sub_path[s])):
                x = np.append(x, sub_path[s][i].pose.position.x)
                y = np.append(y, sub_path[s][i].pose.position.y)

            # Set the interpolation degree
            interp_degree = 5
            if len(x) < 5:
                interp_degree = len(x)-1

            tck, t = interpolate.splprep([x, y], k=interp_degree, s=0)

            bspline.append(tck)

            t = np.linspace(0, 1, num=2000, endpoint=True)
            xy = interpolate.splev(t, tck, der=0)
            dxydt = interpolate.splev(t, tck, der=1)

            arclength = integrate.cumtrapz(np.sqrt(dxydt[0] ** 2 + dxydt[1] ** 2), t, initial=0)

            new_length = 0.0
            prev_length = 0.0
            t_equi = np.array([0])
            for l in zip(t, arclength):
                new_length = l[1] - prev_length
                if new_length >= segment_length:
                    t_equi = np.append(t_equi, l[0])
                    prev_length = l[1]
            t_equi = np.append(t_equi, 1)
            xy_equi = interpolate.splev(t_equi, tck, der=0)

            # Set the reference trajectory for each equidistant points
            ref_trajectory.append(np.zeros((len(t_equi), 5)))
            for i in range(len(t_equi)-1):
                # Evaluate BSpline object
                xy = interpolate.splev(t_equi[i], tck, der=0)
                dxydt = interpolate.splev(t_equi[i], tck, der=1)
                xy_next = interpolate.splev(t_equi[i+1], tck, der=0)
                dxydt_next = interpolate.splev(t_equi[i+1], tck, der=1)

                # Evaluate the distance between two points
                d = sqrt((xy_next[1] - xy[1]) ** 2 + (xy_next[0] - xy[0]) ** 2)

                # Store the pose at each point
                ref_trajectory[-1][i, 0] = xy[0]
                ref_trajectory[-1][i, 1] = xy[1]
                if is_subpath_reversed[s]:
                    # Backward direction
                    phi = wraptopi(np.arctan2(dxydt[1], dxydt[0])+np.pi)
                    phi_next = wraptopi(np.arctan2(dxydt_next[1], dxydt_next[0])+np.pi)
                    delta_phi = phi_next - phi

                    ref_trajectory[-1][i, 2] = phi
                    ref_trajectory[-1][i, 3] = - d / self.Ts
                    ref_trajectory[-1][i, 4] = - delta_phi / self.Ts
                else:
                    # Frontward direction
                    phi = np.arctan2(dxydt[1], dxydt[0])
                    phi_next = np.arctan2(dxydt_next[1], dxydt_next[0])
                    delta_phi = phi_next - phi

                    ref_trajectory[-1][i, 2] = phi
                    ref_trajectory[-1][i, 3] = d / self.Ts
                    ref_trajectory[-1][i, 4] = delta_phi / self.Ts

            # Compute the last point of the trajectory
            xy = interpolate.splev(t_equi[-1], tck, der=0)
            dxydt = interpolate.splev(t_equi[-1], tck, der=1)
            ref_trajectory[-1][-1, 0] = xy[0]
            ref_trajectory[-1][-1, 1] = xy[1]
            if is_subpath_reversed[s]:
                ref_trajectory[-1][-1, 2] = wraptopi(np.arctan2(dxydt[1], dxydt[0]) + np.pi)
            else:
                ref_trajectory[-1][-1, 2] = np.arctan2(dxydt[1], dxydt[0])
            ref_trajectory[-1][-1, 3] = 0.0
            ref_trajectory[-1][-1, 4] = 0.0

            # Plots
            if plotting:
                t = np.linspace(0, 1, num=2000, endpoint=True)
                xy = interpolate.splev(t, tck, der=0)
                dxydt = interpolate.splev(t, tck, der=1)

                plt.figure(2*s)
                plt.plot(x, y, 'bx', xy[0], xy[1], 'b', xy_equi[0], xy_equi[1], 'ro')
                plt.legend(['Received path', 'Interpolated B-spline', 'Equidistant points'], loc='best')
                plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])
                plt.title('B-Spline interpolation')

                if is_subpath_reversed[s]:
                    plt.figure(2*s+1)
                    plt.plot(t, wraptopi(np.arctan2(dxydt[1], dxydt[0])+np.pi) * 180 / np.pi, 'b')
                    plt.title('Phi')
                else:
                    plt.figure(2*s+1)
                    plt.plot(t, np.arctan2(dxydt[1], dxydt[0]) * 180 / np.pi, 'b')
                    plt.title('Phi')

        if plotting:
            plt.show()

        return ref_trajectory
