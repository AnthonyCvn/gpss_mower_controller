#!/usr/bin/env python
import rospy
import tf
import numpy as np
from math import sqrt
from orunav_msgs.srv import ExecuteTask
from orunav_msgs.msg import ControllerReport

from linear_mpc_controller import Controller


class ControllerManager:
    """ ...

    ...

    Attributes:
        ...: ...
    """
    def __init__(self):
        """ ... """
        self.srv_execute_task = None
        self.name_execute_task = 'robot1/execute_task2'
        self.robot_id = 1
        self.controller = Controller()

        self.delta_t = 0.1

    def run(self):
        """ ... """
        rospy.loginfo("Initialize controller services.")
        self.srv_execute_task = rospy.Service(self.name_execute_task, ExecuteTask, self.handle_task_execution)

    def handle_task_execution(self, task):
        """ ... """
        if task.task.abort:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
        else:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE

        self.controller.controller_report.robot_id = self.robot_id

        self.controller.ref_trajectory = self.trajectory_planning(task.task.path.path)
        self.controller.ref_path = self.controller.ref_trajectory[:, 0:3]
        self.controller.path_length = len(task.task.path.path)

    def trajectory_planning(self, path):
        """ ... """
        # Set the trajectory for the controller in a numpy array
        path_len = len(path)
        ref_trajectory = np.zeros((path_len, 5))
        for i in range(path_len-1):
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

            d = sqrt((path[i + 1].pose.position.y - path[i].pose.position.y)**2 +
                     (path[i + 1].pose.position.x - path[i].pose.position.x)**2)

            delta_phi = euler_path_next[2] - euler_path[2]

            ref_trajectory[i, 0] = path[i].pose.position.x
            ref_trajectory[i, 1] = path[i].pose.position.y
            ref_trajectory[i, 2] = euler_path[2]
            ref_trajectory[i, 3] = d / self.delta_t
            ref_trajectory[i, 4] = delta_phi / self.delta_t

        ref_trajectory[-1, 0] = path[-1].pose.position.x
        ref_trajectory[-1, 1] = path[-1].pose.position.y
        ref_trajectory[-1, 2] = euler_path_next[2]
        ref_trajectory[-1, 3] = 0.0
        ref_trajectory[-1, 4] = 0.0

        return ref_trajectory



