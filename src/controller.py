#!/usr/bin/env python
import rospy
import tf
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

from math import pi, atan2, sqrt

DEBUG = False


class Controller:
    """ ...

    ...

    Attributes:
        ...: ...
    """
    def __init__(self):
        # Controller variables
        self.u = np.zeros((2, 1))
        self.i = 0
        self.forward = True

        # Path
        self.path = Path()
        self.goal_i = 0
        self.i_max = 0
        self.seq_change = []
        self.seq_is_reverse = []
        self.idle = True
        self.change_subgoal = True
        self.is_reverse = False

        # ROS variable
        self.goal_pose = Twist()
        self.cmd_vel = Twist()

        # Path subscriber
        rospy.Subscriber("/robot0/global_path", Path, self.path_cb)

        # Command publisher
        self.pub_cmd = rospy.Publisher('/robot0/diff_drive_controller/cmd_vel', Twist, queue_size=1)

    def compute(self, x):
        ## Parameters
        # Lyapunnov stability for: k_rho > 0 ; k_alpha > k_rho ; k_beta < 0
        k_rho = 0.7 / pi
        k_alpha = 1.0 / pi
        k_beta = - 1.0 / pi
        dx = 0.0
        rho = 0.0
        dy = 0.0
        phi = 0.0

        # print "change subgoal: " + str(self.change_subgoal)
        # print self.seq_change

        ## Define next goal
        if not self.idle:
            if self.seq_change and self.change_subgoal:
                self.i_max = self.seq_change.pop(0)
                self.change_subgoal = False

            for i in range(self.goal_i, self.i_max):
                dx = self.path.poses[i].pose.position.x - x[0]
                dy = self.path.poses[i].pose.position.y - x[1]
                rho = sqrt(dx ** 2 + dy ** 2)
                if rho >= 0.4:
                    break

            self.goal_i = i
            self.is_reverse = self.seq_is_reverse[i]

            # print "i max: " + str(self.i_max)
            # print "goal i: " + str(i)
            # print "is reverse: " + str(self.is_reverse)
            # print "rho : " + str(rho)
            # print ""

            quaternion_world2goal = (
                self.path.poses[i].pose.orientation.x,
                self.path.poses[i].pose.orientation.y,
                self.path.poses[i].pose.orientation.z,
                self.path.poses[i].pose.orientation.w)
            euler_world2goal = tf.transformations.euler_from_quaternion(quaternion_world2goal, axes='sxyz')
            phi = euler_world2goal[2]

        ## Go to the goal
        if rho < 0.01:
            self.change_subgoal = True
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
        else:
            theta = x[2]
            alpha = self.wraptopi(atan2(dy, dx) - theta)
            beta = self.wraptopi(phi - alpha - theta)

            if self.is_reverse:
                self.cmd_vel.angular.z = k_alpha * self.wraptopi(alpha - pi) + k_beta * self.wraptopi(beta - pi)
                self.cmd_vel.linear.x = - k_rho * rho
            else:
                self.cmd_vel.angular.z = k_alpha * alpha + k_beta * beta
                self.cmd_vel.linear.x = k_rho * rho

        self.pub_cmd.publish(self.cmd_vel)

        self.u[0] = self.cmd_vel.linear.x
        self.u[1] = self.cmd_vel.angular.z

        if DEBUG:
            print "rho"
            print rho
            print "alpha"
            print alpha / pi * 180
            print "beta"
            print beta / pi * 180
            print "theta"
            print x[2] / pi * 180

        return self.u

    def path_cb(self, path):
        self.path = path
        self.goal_i = 0
        self.idle = False
        self.change_subgoal = True
        self.seq_change = []
        self.seq_is_reverse = []

        if len(self.path.poses) > 3:
            # Detect change of direction
            for i in range(len(self.path.poses) - 2):
                byaw0 = atan2(self.path.poses[i + 1].pose.position.y - self.path.poses[i].pose.position.y,
                              self.path.poses[i + 1].pose.position.x - self.path.poses[i].pose.position.x)
                byaw1 = atan2(self.path.poses[i + 2].pose.position.y - self.path.poses[i + 1].pose.position.y,
                              self.path.poses[i + 2].pose.position.x - self.path.poses[i + 1].pose.position.x)
                if abs(self.wraptopi(byaw1 - byaw0)) > pi / 2:
                    self.seq_change.append(i + 1)

            for i in range(len(self.path.poses) - 1):
                byaw0 = atan2(self.path.poses[i + 1].pose.position.y - self.path.poses[i].pose.position.y,
                              self.path.poses[i + 1].pose.position.x - self.path.poses[i].pose.position.x)

                quaternion_world2goal = (
                    self.path.poses[i].pose.orientation.x,
                    self.path.poses[i].pose.orientation.y,
                    self.path.poses[i].pose.orientation.z,
                    self.path.poses[i].pose.orientation.w)
                euler_world2goal = tf.transformations.euler_from_quaternion(quaternion_world2goal, axes='sxyz')
                phi = euler_world2goal[2]

                if abs(self.wraptopi(phi - byaw0)) > pi / 2:
                    self.seq_is_reverse.append(True)
                else:
                    self.seq_is_reverse.append(False)
            self.seq_is_reverse.append(self.seq_is_reverse[-1])

            for i in self.seq_change:
                self.seq_is_reverse[i] = self.seq_is_reverse[i - 1]

            self.i_max = len(self.path.poses) - 1
            self.seq_change.append(self.i_max)
