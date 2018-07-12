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

# Specific controller's libraries.
from linear_mpc_controller import Controller
from toolbox import wraptopi


class TaskManager:
    """ Handle task execution of the robot.

    Receive a task and transmit the information to the controller:
        1) Extract the path from ExecuteTask message.
        2) Plan a trajectory for the mobile robot.
        3) Send the trajectory to the controller.
        2) Change the controller's status to active mode.

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
        2) Change the controller's status to active mode.

        """
        self.controller.controller_report.robot_id = self.robot_id
        self.controller.ref_trajectory = self.trajectory_planning(task.task.path.path)
        self.controller.ref_path = self.controller.ref_trajectory[:, 0:3]
        self.controller.path_length = len(task.task.path.path)

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
        # Set the trajectory for the controller in a numpy array
        path_len = len(path)
        ref_trajectory = np.zeros((path_len, 5))
        for i in range(path_len-1):
            # Compute distance 'd' and delta angle 'delta_phi_pose' between two consecutive points on the path
            quaternion_path = (
                path[i].pose.orientation.x,
                path[i].pose.orientation.y,
                path[i].pose.orientation.z,
                path[i].pose.orientation.w)
            euler_path = tf.transformations.euler_from_quaternion(quaternion_path, axes='sxyz')
            quaternion_path_next = (
                path[i+1].pose.orientation.x,
                path[i+1].pose.orientation.y,
                path[i+1].pose.orientation.z,
                path[i+1].pose.orientation.w)
            euler_path_next = tf.transformations.euler_from_quaternion(quaternion_path_next, axes='sxyz')

            phi_pose = euler_path[2]
            delta_phi_pose = euler_path_next[2] - phi_pose

            d = sqrt((path[i + 1].pose.position.y - path[i].pose.position.y)**2 +
                     (path[i + 1].pose.position.x - path[i].pose.position.x)**2)

            # Compute the delta angle 'delta_phi_path' between two consecutive points on the path
            delta_phi_path = atan2(path[i + 1].pose.position.y - path[i].pose.position.y,
                                   path[i + 1].pose.position.x - path[i].pose.position.x)

            # Compute the reference trajectory
            ref_trajectory[i, 0] = path[i].pose.position.x
            ref_trajectory[i, 1] = path[i].pose.position.y
            ref_trajectory[i, 2] = euler_path[2]
            if abs(wraptopi(delta_phi_path - phi_pose)) > pi / 2:
                # Backward direction
                ref_trajectory[i, 3] = - d / self.Ts
                ref_trajectory[i, 4] = - delta_phi_pose / self.Ts
            else:
                # Frontward direction
                ref_trajectory[i, 3] = d / self.Ts
                ref_trajectory[i, 4] = delta_phi_pose / self.Ts

        # Compute the last point of the trajectory
        ref_trajectory[-1, 0] = path[-1].pose.position.x
        ref_trajectory[-1, 1] = path[-1].pose.position.y
        ref_trajectory[-1, 2] = euler_path_next[2]
        ref_trajectory[-1, 3] = 0.0
        ref_trajectory[-1, 4] = 0.0

        return ref_trajectory



