#!/usr/bin/env python

# ROS libraries for Python.
import rospy

# Specific controller's libraries.
from task_receptor import TaskReceptor
from tf_mng import TfMng
from estimator import Estimator
import ltv_mpc


def main():
    rospy.init_node("gpss_mower_controller")

    # Read and set parameters.
    robot_id = rospy.get_param("~robot_id")
    sampling_period = 1.0 / rospy.get_param("~sampling_frequency")
    horizon = rospy.get_param("~horizon")
    desire_speed = rospy.get_param("~desire_speed")
    deceleration_distance = rospy.get_param("~deceleration_distance")
    photo_activated = rospy.get_param("~photo_activated")
    print_status = rospy.get_param("~print_status")
    log_to_file = rospy.get_param("~log_to_file")
    max_speed = rospy.get_param("~max_forward_velocity")
    max_ang_speed = rospy.get_param("~max_angular_velocity")
    max_acc = rospy.get_param("~max_tan_acceleration")
    weight_x = rospy.get_param("~weight_x")
    weight_y = rospy.get_param("~weight_y")
    weight_theta = rospy.get_param("~weight_theta")
    weight_v = rospy.get_param("~weight_v")
    weight_w = rospy.get_param("~weight_w")

    min_speed = - max_speed
    min_ang_speed = -max_ang_speed
    min_acc = -max_acc
    mpc_weights = [weight_x, weight_y, weight_theta, weight_v, weight_w]
    constraints = [max_speed, min_speed, max_ang_speed, min_ang_speed, max_acc, min_acc]

    # Run the controller
    regulator = ltv_mpc.Regulator(robot_id, sampling_period, horizon, mpc_weights, constraints)

    TaskReceptor(robot_id, regulator, desire_speed, deceleration_distance)

    tf_manager = TfMng(robot_id, photo_activated)
    if photo_activated:
        Estimator(robot_id, regulator, tf_manager, Estimator.EKF_RECALCULATION, print_status, log_to_file)
    else:
        Estimator(robot_id, regulator, tf_manager, Estimator.EKF_ODOM, print_status, log_to_file)

    # Blocks until ROS node is shutdown.
    rospy.spin()


if __name__ == "__main__":
    main()
