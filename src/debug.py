#!/usr/bin/env python
import luts
import toolbox


def debug():
    print"|------------------------- Test LUTs -------------------------|"
    tf_lut = luts.TfLut()
    t = toolbox.TicToc()
    t.tic()
    a = tf_lut.T_world2cam[0]
    print "Time to access T_world2cam = " + str(t.toc())
    print "T_world2cam: " + str(a)


if __name__ == "__main__":
    debug()