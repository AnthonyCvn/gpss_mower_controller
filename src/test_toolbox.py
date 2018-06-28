#!/usr/bin/env python
import unittest

from toolbox import *
from math import pi

## A sample python unit test
class TestBareBones(unittest.TestCase):

    def test_wraptopi(self):
        a = pi/2
        b = wraptopi(-3*pi/2)
        self.assertEquals(a, b, "wraptopi error")


if __name__ == '__main__':
    import rosunit
    rosunit.unitrun("gpss_mower_controller", 'test_bare_bones', TestBareBones)
