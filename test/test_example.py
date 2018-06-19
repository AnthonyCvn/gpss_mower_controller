#!/usr/bin/env python
import unittest
from math import pi

from toolbox import wraptopi

## A sample python unit test
class TestBareBones(unittest.TestCase):

    def test_one_equals_one(self):
        self.assertEquals(1, 1, "1!=1")

    def test_wraptopi(self):
        a = pi/2
        b = wraptopi(-3*pi/2)
        self.assertEquals(a, b, "1!=1")


if __name__ == '__main__':
    import rosunit
    rosunit.unitrun("gpss_mower_controller", 'test_bare_bones', TestBareBones)
