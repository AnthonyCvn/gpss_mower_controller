#!/usr/bin/env python
import rospy

from filter import Filter
from controller import Controller


def main():
    rospy.init_node("gpss_mower_controller")

    controller = Controller()

    filter = Filter()
    filter.ctrl = controller
    filter.Ts = 0.1

    filter.run()

    # Blocks until ROS node is shutdown.
    rospy.spin()

if __name__ == "__main__":
    main()