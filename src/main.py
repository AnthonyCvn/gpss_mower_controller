#!/usr/bin/env python

# ROS libraries for Python.
import rospy

# ROS messages.
from geometry_msgs.msg import Twist

# Specific controller's libraries.
from task_manager import TaskManager
from tf_mng import TfMng
from filter import Filter
from linear_mpc_controller import Controller

MIR100_SIM = False
ORU_SIM = True

def main():
    rospy.init_node("gpss_mower_controller")

    # Parameters.
    Ts = 0.1
    horizon = 15
    robot_id = 2

    # Create a controller, a controller manager and a filter object.
    controller = Controller(Ts, horizon)

    controller_manager = TaskManager()
    controller_manager.controller = controller
    controller_manager.robot_id = robot_id

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
    controller_manager.run()

    # Blocks until ROS node is shutdown.
    rospy.spin()

if __name__ == "__main__":
    main()