#!/usr/bin/env python

# ROS libraries for Python.
import rospy
import tf
import message_filters

# ROS messages.
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

# Python packages.
import numpy as np
from scipy import linalg
from math import atan2

# Specific controller's libraries.
from toolbox import Sensors


class TfMng:
    """ The transform manager store the sensors' data and send the TF transform between /map and /odom..

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

    def __init__(self, robot_id, photo_activated):
        self.robot_id = 1
        self.photo_activated = photo_activated
        self.odom_frame_id = "/robot{0}/odom".format(robot_id)
        self.odom_topic = "/robot{0}/odom".format(robot_id)
        self.photo_topic = "/world_tags/hrp{0}{1}".format(robot_id/10, robot_id % 10)
        self.map_frame_id = "/map"

        self.elevation_frame = 0.12

        self.odom_pose = np.zeros((3, 1))
        self.photo_pose = np.zeros((3, 1))

        self.T_world2odom = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_odom2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.odom_world2robot = np.zeros((3, 1))

        self.odom_odom2robot = Odometry()
        self.photo_world2robot = PoseStamped()

        self.br = tf.TransformBroadcaster()

        self.sensors = Sensors()

        # Start the transform manager by subscribing to the desire sensors' topics.

        if self.photo_activated:
            rospy.loginfo("Localization of Robot #{0} is based on odometry and photogrammetry.".format(self.robot_id))
            rospy.Subscriber(self.photo_topic, PoseStamped, self.photo_cb)
            rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

            # Synchronize topics with message_filter.
            photo_sub = message_filters.Subscriber(self.photo_topic, PoseStamped)
            odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
            ts = message_filters.ApproximateTimeSynchronizer([odom_sub, photo_sub], 20, 0.05, allow_headerless=False)
            ts.registerCallback(self.odom_photo_synch_cb)

        else:
            rospy.loginfo("Localization of Robot #{0} is only based on odometry.".format(self.robot_id))
            rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

    def photo_cb(self, photo):
        """ Callback function when sensed value comes from the cameras."""

        self.photo_world2robot = photo

        self.photo_pose = self.get_photo_pose(photo)

        self.sensors.is_photo = True
        self.sensors.photo_pose = self.get_photo_pose(photo)

        #self.sensors.photo_t = photo.header.stamp.to_sec()
        #print"Photo delay: ", abs(self.sensors.photo_t - rospy.get_rostime().now().to_sec())

    def odom_cb(self, odom):
        """ Callback function when sensed value comes from odometry.

        Store the pose of the robot relative to the /world coordinate according to the odometry feedback.

        Args:
            odom (Odometry): Pose of the robot (base_footprint) relative to /odom coordinate.

        Update:
            sensors: Pose and time stamp of the odometry relative to the /world coordinate.
        """
        self.odom_odom2robot = odom

        self.odom_pose = self.get_world2robot(odom)

        self.sensors.odom_pose = self.get_world2robot(odom)
        self.sensors.odom_t = odom.header.stamp.to_sec()

        # print"odom delay: ", abs(self.sensors.odom_t - rospy.get_rostime().now().to_sec())

    def odom_photo_synch_cb(self, odom, photo):

        filename = "/home/anthony/log_sensors.txt"

        odom_pose = self.get_world2robot(odom)
        photo_pose = self.get_photo_pose(photo)

        if True:
            with open(filename, "a+") as my_file:
                my_file.write("{0},{1},{2},{3},{4},{5},{6},{7}\n"
                              .format(odom_pose[0][0], odom_pose[1][0], odom_pose[2][0], odom.header.stamp.to_sec(),
                                      photo_pose[0][0], photo_pose[1][0], photo_pose[2][0], photo.header.stamp.to_sec()))

    @staticmethod
    def get_photo_pose(photo):
        photo_world2robot = np.zeros((3, 1))

        quaternion_photo = (photo.pose.orientation.x,
                            photo.pose.orientation.y,
                            photo.pose.orientation.z,
                            photo.pose.orientation.w)

        euler_rotation = tf.transformations.euler_from_quaternion(quaternion_photo, axes='sxyz')

        photo_world2robot[0] = photo.pose.position.x
        photo_world2robot[1] = photo.pose.position.y
        photo_world2robot[2] = euler_rotation[2]

        return photo_world2robot

    def get_world2robot(self, odom):
        """ Get the transform between /world coordinate and /robot coordinate by knowing /odom-to-/robot transform.

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
        self.T_odom2robot[2, 3] = odom.pose.pose.position.z - self.elevation_frame

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
                              self.map_frame_id)



