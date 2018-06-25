#!/usr/bin/env python
import rospy

from controller import Controller
from tf_mng import TfMng
from filter import Filter

from geometry_msgs.msg import Twist

SIMULATION = False


def main():
    rospy.init_node("gpss_mower_controller")

    controller = Controller()
    tf_manager = TfMng()

    if SIMULATION:
        controller.pub_cmd = rospy.Publisher('/robot0/diff_drive_controller/cmd_vel', Twist, queue_size=1)
        tf_manager.odom_frame_id = "robot0/odom"
        tf_manager.odom_topic = "robot0/diff_drive_controller/odom"

    mower_filter = Filter()
    mower_filter.Ts = 0.1
    mower_filter.ctrl = controller
    mower_filter.tf_mng = tf_manager

    mower_filter.run()

    # Blocks until ROS node is shutdown.
    rospy.spin()

if __name__ == "__main__":
    main()