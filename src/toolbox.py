#!/usr/bin/env python
import time
from math import pi
import numpy as np


def wraptopi(angle):
    """ Wrap angle between -pi and pi. """
    angle = (angle + pi) % (2 * pi) - pi
    return angle


class Sensors:
    def __init__(self):
        self.odom_pose = np.zeros((3, 1))
        self.marker_pose = np.zeros((3, 1))
        self.t = 0.0


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