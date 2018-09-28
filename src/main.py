#!/usr/bin/env python

# ROS libraries for Python.
import rospy

# ROS messages.
from geometry_msgs.msg import Twist
from orunav_msgs.msg import ControllerReport

# Specific controller's libraries.
from task_manager import TaskManager
from tf_mng import TfMng
from filter import Filter
from linear_mpc_controller import Controller


def main():
    rospy.init_node("gpss_mower_controller")

    # Get parameters.
    robot_id = rospy.get_param("~robot_id")
    sampling_period = 1.0 / rospy.get_param("~sampling_frequency")
    horizon = rospy.get_param("~horizon")
    desire_speed = rospy.get_param("~desire_speed")
    compensate_delay = rospy.get_param("~compensate_delay")
    max_forward_velocity = rospy.get_param("~max_forward_velocity")
    max_angular_velocity = rospy.get_param("~max_angular_velocity")
    max_tan_acceleration = rospy.get_param("~max_tan_acceleration")
    weight_x = rospy.get_param("~weight_x")
    weight_y = rospy.get_param("~weight_y")
    weight_theta = rospy.get_param("~weight_theta")
    weight_v = rospy.get_param("~weight_v")
    weight_w = rospy.get_param("~weight_w")

    # Create a controller, a task manager and a sensors_filter object.
    mpc_weights = [weight_x, weight_y, weight_theta, weight_v, weight_w]
    controller = Controller(robot_id, sampling_period, horizon, mpc_weights)
    controller.u_max[0] = max_forward_velocity
    controller.u_max[1] = max_angular_velocity
    controller.u_min[0] = -max_forward_velocity
    controller.u_min[1] = -max_angular_velocity
    controller.a_tan_max = max_tan_acceleration
    controller.a_tan_min = -max_tan_acceleration

    task_manager = TaskManager()
    task_manager.controller = controller
    task_manager.a_tan_max = max_tan_acceleration
    task_manager.robot_id = robot_id
    task_manager.desire_speed = desire_speed
    task_manager.task_srv_name = "robot{0}/execute_task2".format(robot_id)
    task_manager.executer_command_topic = "robot{0}/controller/commands".format(robot_id)
    task_manager.executer_trajectories_topic = "robot{0}/controller/trajectories".format(robot_id)
    task_manager.Ts = sampling_period

    tf_manager = TfMng()
    tf_manager.robot_id = robot_id
    tf_manager.odom_frame_id = "/robot{0}/odom".format(robot_id)
    tf_manager.odom_topic = "/robot{0}/odom".format(robot_id)
    tf_manager.photogrammetry_topic = "/world_tags/hrp{0}{1}".format(robot_id/10, robot_id % 10)
    tf_manager.world_frame_id = "/map"
    tf_manager.photo_activated = False


    sensors_filter = Filter()
    sensors_filter.robot_id = robot_id
    sensors_filter.Ts = sampling_period
    sensors_filter.ctrl = controller
    sensors_filter.tf_mng = tf_manager
    sensors_filter.pub_cmd = rospy.Publisher("/robot{0}/cmd_vel".format(robot_id), Twist, queue_size=1)
    sensors_filter.compensate_delay = compensate_delay

    controller.pub_robot_report = rospy.Publisher('/robot{0}/controller/reports'.format(robot_id), ControllerReport, queue_size=1)
    controller.pub_robot_pose = rospy.Publisher('/robot{0}/pose_estimate'.format(robot_id), Twist, queue_size=1)
    
    # Launch the program.
    sensors_filter.run()
    task_manager.run()

    # Blocks until ROS node is shutdown.
    rospy.spin()


if __name__ == "__main__":
    main()