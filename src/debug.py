#!/usr/bin/env python
import luts
import toolbox

import numpy as np
from scipy import interpolate
from scipy import spatial
import matplotlib.pyplot as plt


def debug():
    print"|------------------------- Test LUTs -------------------------|"
    tf_lut = luts.TfLut()
    t = toolbox.TicToc()
    t.tic()
    a = tf_lut.T_world2cam[0]
    print "Time to access T_world2cam = " + str(t.toc())
    print "T_world2cam: " + str(a)
    print""

    print"|------------------------- Test Spline -------------------------|"
    x = np.arange(0, 2 * np.pi + np.pi / 4, 2 * np.pi / 8)
    y = np.sin(x)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, 2 * np.pi, np.pi / 50)
    ynew = interpolate.splev(xnew, tck, der=0)
    plt.figure()
    plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
    plt.legend(['Linear', 'Cubic Spline', 'True'])
    plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title('Cubic-spline interpolation')
    plt.show()
    print""

    print"|------------------------- Test KDTree -------------------------|"
    A = np.random.random((100, 2)) * 100
    pt = [6, 30]
    t = toolbox.TicToc()
    t.tic()
    distance, index = spatial.KDTree(A).query(pt)
    elapsed = t.toc()
    nearest = A[spatial.KDTree(A).query(pt)[1]]
    print "A = "+str(A)
    print "point = " + str(pt)
    print "index = " + str(index)
    print "A[index] = "+str(A[index])
    print "Time for KDTree = " + str(elapsed*1000) + " ms"

if __name__ == "__main__":
    debug()