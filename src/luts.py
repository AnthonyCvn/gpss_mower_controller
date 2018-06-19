#!/usr/bin/env python
import tf
from math import pi


class TfLut:
    """Store transformation between /World frame and Cameras in a LUT style.

    Each cameras are fixed and shouldn't move after the calibration.
    The transforms between cameras and the World frame are stored in homogeneous matrix format.
    For example, first create the TF lookup table with: tf_lut = luts.TfLut()
    Then, the 4x4 homogeneous matrix transform for the camera ID 0 is given by: tf_lut.T_world2cam[0]

    Attributes:
        T_world2cam: 4x4 homogeneous matrices stored as a lookup table.
    """
    def __init__(self):
        """ Define transformation matrices for each cameras in initialization """
        # Transform between World and Camera ID 0
        self.T_world2cam = []
        self.T_world2cam.append(tf.transformations.euler_matrix(0, pi, pi / 2, 'sxyz'))
        self.T_world2cam[0][2, 3] = 2.50




