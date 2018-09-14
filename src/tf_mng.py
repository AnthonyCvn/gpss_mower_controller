#!/usr/bin/env python

# ROS libraries for Python.
import rospy
import tf

# ROS messages.
from nav_msgs.msg import Odometry

# Python packages.
import numpy as np
from scipy import linalg
from math import atan2

# Specific controller's libraries.
from toolbox import Sensors

class TfMng:
    """ The transform manager receive, process and store the sensors values.

    The transform manager receive, process and store the sensors values as following:
    1)  Receive the sensors values on different callback functions.
    2)  Transform the sensed values relative to the reference frame coordinate.
    3)  Store transformed values in a buffer.
    4)  Send the tf transform between /world and /odom.

    Note:
        The transform manager runs at sensors frequency.

    Attributes:
        photo_activated     : True if sensed values are odometry and photogrametry. False if only based on odometry.
        T_world2odom        : Homogeneous transform between /world coordinate and /odom coordinate.
        T_odom2robot        : Homogeneous transform between /odom coordinate and /robot coordinate.
        T_world2robot       : Homogeneous transform between /world coordinate and /robot coordinate
        odom_world2robot    : Homogeneous transform (/world to /robot) based on odometry.
        dim_z               : Dimension of the z vector.
        z                   : Sensed values relative to /world coordinate storage.

    """

    def __init__(self):
        self.photo_activated = False

        self.T_world2odom = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_odom2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.odom_world2robot = np.zeros((3, 1))

        self.robot_pose = np.zeros((3, 1))

        self.br = tf.TransformBroadcaster()

        self.sensors = Sensors()

        self.robot_id = 1

        self.odom_frame_id = "/odom"
        self.odom_topic = "/odom"

        self.world_frame_id = "/world"

    def run(self):
        """ Start the transform manager by subscribing to the desire sensors' topics. """

        if self.photo_activated:
            rospy.loginfo("Localization of Robot #{0} is based on odometry and photogrammetry.".format(self.robot_id))
        else:
            rospy.loginfo("Localization of Robot #{0} is only based on odometry.".format(self.robot_id))
            rospy.Subscriber(self.odom_topic, Odometry, self.odometry_only_cb)

    def odometry_only_cb(self, odom):
        """ Callback function when sensed value is only odometry.

        Store the pose of the robot relative to the /world coordinate according to the odometry feedback.

        Args:
            odom (Odometry): Pose of the robot (base_footprint) relative to /odom coordinate.

        Update:
            sensors: Pose and time stamp of the odometry relative to the /world coordinate.
        """

        self.sensors.odom_pose = self.get_world2robot(odom)
        self.sensors.t = odom.header.stamp.to_sec()
        #print abs(self.sensors.t - rospy.get_rostime().now().to_sec())

    def get_world2robot(self, odom):
        """ Get the transform between /world coordinate and /robot coordinate by knowing /odom to /robot transform.

        Args:
            odom (Odometry): Pose of the robot relative to /odom coordinate.

        Returns:
            odom_world2robot: numpy vector (x, y, phi) with the pose of robot relative to /world coordinate.

        Updates:
            T_world2odom: Homogeneous transform between /world coordinate and /odom coordinate.
            T_odom2robot: Homogeneous transform between /odom coordinate and /robot coordinate.
        """
        quaternion_odom2robot = (odom.pose.pose.orientation.x,
                                 odom.pose.pose.orientation.y,
                                 odom.pose.pose.orientation.z,
                                 odom.pose.pose.orientation.w)

        self.T_odom2robot = tf.transformations.quaternion_matrix(quaternion_odom2robot)
        self.T_odom2robot[0, 3] = odom.pose.pose.position.x
        self.T_odom2robot[1, 3] = odom.pose.pose.position.y
        self.T_odom2robot[2, 3] = odom.pose.pose.position.z

        self.T_world2robot = self.T_world2odom.dot(self.T_odom2robot)

        self.odom_world2robot[0] = self.T_world2robot[0, 3]
        self.odom_world2robot[1] = self.T_world2robot[1, 3]
        self.odom_world2robot[2] = atan2(self.T_world2robot[1, 0], self.T_world2robot[0, 0])

        return self.odom_world2robot

    def update_world2odom(self, robot_pose, T_odom2robot):
        """ Update the transform between /world coordinate and /odom coordinate.

        Note:
            Send the tf transform between /world and /odom.

        Args:
            robot_pose   : Pose of the robot relative to /world coordinate.
            T_odom2robot : Specific transform between /odom and /robot coordinate.

        Returns:
            odom_world2robot: numpy vector (x, y, phi) with the pose of robot relative to /world coordinate.

        Update:
            T_world2odom: Homogeneous transform between /world coordinate and /odom coordinate.
        """
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, robot_pose[2], 'sxyz')
        self.T_world2robot[0, 3] = robot_pose[0]
        self.T_world2robot[1, 3] = robot_pose[1]
        self.T_world2odom = self.T_world2robot.dot(linalg.inv(T_odom2robot))

        quaternion_world2odom = tf.transformations.quaternion_from_matrix(self.T_world2odom)
        self.br.sendTransform(self.T_world2odom[0:3, 3],
                              quaternion_world2odom,
                              rospy.Time.now(),
                              self.odom_frame_id,
                              self.world_frame_id)