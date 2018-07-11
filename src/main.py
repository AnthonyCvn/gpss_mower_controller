#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

from controller_mng import ControllerManager
from tf_mng import TfMng
from filter import Filter

from linear_mpc_controller import Controller

MIR100_SIM = False
ORU_SIM = True


def main():
    rospy.init_node("gpss_mower_controller")

    Ts = 0.1
    horizon = 15
    robot_id = 1

    controller = Controller(Ts, horizon)
    tf_manager = TfMng()
    mower_filter = Filter()
    mower_filter.Ts = Ts
    mower_filter.ctrl = controller
    mower_filter.tf_mng = tf_manager

    if MIR100_SIM:
        mower_filter.pub_cmd = rospy.Publisher('/robot0/diff_drive_controller/cmd_vel', Twist, queue_size=1)
        tf_manager.odom_frame_id = "robot0/odom"
        tf_manager.odom_topic = "robot0/diff_drive_controller/odom"

    if ORU_SIM:
        mower_filter.pub_cmd = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=1)
        tf_manager.odom_frame_id = "/robot1/odom"
        tf_manager.odom_topic = "/robot1/odom"

    mower_filter.run()

    controller_manager = ControllerManager()
    controller_manager.controller = controller
    controller_manager.robot_id = robot_id
    controller_manager.run()

    # Blocks until ROS node is shutdown.
    rospy.spin()

if __name__ == "__main__":
    main()