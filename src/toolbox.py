#!/usr/bin/env python

# Python packages.
import time
from math import pi
import numpy as np

# ROS messages.
from geometry_msgs.msg import Twist


def wraptopi(angle):
    """ Wrap angle between -pi and pi. """
    angle = (angle + pi) % (2 * pi) - pi
    return angle

def twist_pose_pub(self, numpy_pose, twist_publisher):
    """ Publish a numpy array that describe a pose (x, y, phi) on a Twist ROS message. """
    twist_pose = Twist()
    twist_pose.linear.x = numpy_pose[0]
    twist_pose.linear.y = numpy_pose[1]
    twist_pose.angular.z = numpy_pose[2]
    twist_publisher.publish(twist_pose)

class Sensors:
    """ Store sensors value and timestamp. """
    def __init__(self):
        self.odom_pose = np.zeros((3, 1))
        self.odom_t = 0.0
        self.photo_pose = np.zeros((3, 1))
        self.photo_t = 0.0
        self.is_photo = False


class TicToc:
    """ Time measurement object.

    Measure time as following:
    t = TicToc()
    t.tic()
    time.sleep(2)
    print t.toc()
    2.00210309029

    """
    def __init__(self):
        self.stack = []
        self.named = {}
        self.elapsed = 0

    def tic(self, name=None):
        if name is None:
            self.stack.append(time.time())
        else:
            self.named[name] = time.time()

    def toc(self, name=None):
        if name is None:
            start = self.stack.pop()
        else:
            start = self.named.pop(name)
        self.elapsed = time.time() - start
        return self.elapsed

    def set(self, value, name=None):
        if name is None:
            self.stack[-1] = value
        else:
            self.named[name] = value

    def get(self, name=None):
        if name is None:
            return self.stack[-1]
        else:
            return self.named[name]