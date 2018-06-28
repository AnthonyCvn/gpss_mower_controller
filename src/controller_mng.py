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
        self.srv_execute_task = rospy.Service(self.name_execute_task, ExecuteTask, self.handle_execute_task)

    def handle_execute_task(self, task):
        if task.task.abort:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
        else:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE

        self.controller.controller_report.robot_id = self.robot_id

        self.controller.path = task.task.path

        # Set the trajectory for the controller in a numpy array
        path_len = len(self.controller.path.path)
        self.controller.ref_trajectory = np.zeros((path_len, 5))
        for i in range(path_len-1):
            quaternion_path = (
                task.task.path.path[i].pose.orientation.x,
                task.task.path.path[i].pose.orientation.y,
                task.task.path.path[i].pose.orientation.z,
                task.task.path.path[i].pose.orientation.w)
            euler_path = tf.transformations.euler_from_quaternion(quaternion_path, axes='sxyz')
            quaternion_path_next = (
                task.task.path.path[i+1].pose.orientation.x,
                task.task.path.path[i+1].pose.orientation.y,
                task.task.path.path[i+1].pose.orientation.z,
                task.task.path.path[i+1].pose.orientation.w)
            euler_path_next = tf.transformations.euler_from_quaternion(quaternion_path_next, axes='sxyz')

            d = sqrt((task.task.path.path[i + 1].pose.position.y - task.task.path.path[i].pose.position.y)**2 +
                     (task.task.path.path[i + 1].pose.position.x - task.task.path.path[i].pose.position.x)**2)

            delta_phi = euler_path_next[2] - euler_path[2]

            self.controller.ref_trajectory[i, 0] = task.task.path.path[i].pose.position.x
            self.controller.ref_trajectory[i, 1] = task.task.path.path[i].pose.position.y
            self.controller.ref_trajectory[i, 2] = euler_path[2]
            self.controller.ref_trajectory[i, 3] = d / self.delta_t
            self.controller.ref_trajectory[i, 4] = delta_phi / self.delta_t

        self.controller.ref_trajectory[-1, 0] = task.task.path.path[-1].pose.position.x
        self.controller.ref_trajectory[-1, 1] = task.task.path.path[-1].pose.position.y
        self.controller.ref_trajectory[-1, 2] = euler_path_next[2]
        self.controller.ref_trajectory[-1, 3] = 0.0
        self.controller.ref_trajectory[-1, 4] = 0.0

        print self.controller.ref_trajectory



