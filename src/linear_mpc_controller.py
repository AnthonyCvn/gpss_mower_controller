#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist

from orunav_msgs.msg import ControllerReport
from orunav_msgs.msg import Path


class Controller:
    """ ...

    ...

    Attributes:
        ...: ...
    """
    def __init__(self):
        """ ... """
        # Map the controller status to the corresponding function.
        self.ctr_states = {ControllerReport.CONTROLLER_STATUS_ACTIVE:    self.controller_status_active,
                           ControllerReport.CONTROLLER_STATUS_FAIL:      self.controller_status_fail,
                           ControllerReport.CONTROLLER_STATUS_FINALIZE:  self.controller_status_finalize,
                           ControllerReport.CONTROLLER_STATUS_TERMINATE: self.controller_status_terminate,
                           ControllerReport.CONTROLLER_STATUS_WAIT:      self.controller_status_wait,
                           }

        # Controller variables
        self.u = np.zeros((2, 1))
        self.mu = np.zeros((3, 1))

        # ROS variables
        self.cmd_vel = Twist()
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Controller report
        self.pub_robot_report = rospy.Publisher('/robot1/controller/reports', ControllerReport, queue_size=1)
        self.controller_report = ControllerReport()
        self.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT

        # Reference path and trajectory
        self.path = Path()
        self.ref_trajectory = None

    def compute(self):
        """ ... """
        # Call states' function
        self.ctr_states[self.controller_report.status]()

        # Publish controller command
        self.cmd_vel.linear.x = self.u[0]
        self.cmd_vel.angular.z = self.u[1]
        self.pub_cmd.publish(self.cmd_vel)

        # Publish controller report
        self.controller_report.state.position_x = self.mu[0]
        self.controller_report.state.position_y = self.mu[1]
        self.controller_report.state.orientation_angle = self.mu[2]
        self.controller_report.stamp = rospy.get_rostime()
        self.pub_robot_report.publish(self.controller_report)

    def trajectory_selector(self):
        print self.path.path[0].pose.position.x
        print self.path.path[0].pose.orientation.x

    def controller_status_active(self):
        """ ... """
        self.trajectory_selector()

    def controller_status_fail(self):
        """ ... """
        print"controller_status_fail"

    def controller_status_finalize(self):
        """ ... """
        print"controller_status_finalize("

    def controller_status_terminate(self):
        """ ... """
        print"controller_status_terminate"

    def controller_status_wait(self):
        """ ... """
        print"controller_status_wait"
