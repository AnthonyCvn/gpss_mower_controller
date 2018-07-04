#!/usr/bin/env python
import rospy
import tf
from nav_msgs.msg import Odometry

import numpy as np
from scipy import linalg
from math import atan2

from toolbox import Sensors

class TfMng:
    """ ...

    ...

    Attributes:
        fidmarker_activated : True if Ometry and fiducial markers available. False if only Odometry is available.
        T_world2odom        :
        T_odom2robot        :
        T_world2robot       :
        odom_world2robot    :
        dim_z               :
        z                   :

    """

    def __init__(self):
        """ ... """
        self.fid_marker_activated = False

        self.T_world2odom = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_odom2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.odom_world2robot = np.zeros((3, 1))

        self.robot_pose = np.zeros((3, 1))

        self.br = tf.TransformBroadcaster()

        self.sensors = Sensors()

        self.odom_frame_id = "/odom"
        self.odom_topic = "/odom"
        self.world_frame_id = "/world"

    def run(self):
        """ ... """

        if self.fid_marker_activated:
            rospy.loginfo("Localization based on odometry and fiducial Markers.")
        else:
            rospy.loginfo("Localization based on odometry only.")
            rospy.Subscriber(self.odom_topic, Odometry, self.odometry_only_cb)

    def odometry_only_cb(self, odom):
        """Odometry callback function.

        Store the global pose of the robot with the odometer feedback.

        Args:
            odom (Odometry): Pose of the robot (base_footprint) relative to robot0/odom coordinate.
        """

        self.sensors.odom_pose = self.get_world2robot(odom)
        self.sensors.t = odom.header.stamp.to_sec()

    def get_world2robot(self, odom):
        """Get transforms from /world coordinate to /robot0 coordinate with odometry.

        Args:
            odom (Odometry): Pose of the robot relative to robot0/odom coordinate.

        Returns:
            odom_world2robot: numpy vector (x, y, phi) with the pose of robot0 relative to /world coordinate.
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

    def update_world2odom(self, robot_pose):
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, robot_pose[2], 'sxyz')
        self.T_world2robot[0, 3] = robot_pose[0]
        self.T_world2robot[1, 3] = robot_pose[1]
        self.T_world2odom = self.T_world2robot.dot(linalg.inv(self.T_odom2robot))

        quaternion_world2odom = tf.transformations.quaternion_from_matrix(self.T_world2odom)
        self.br.sendTransform(self.T_world2odom[0:3, 3],
                              quaternion_world2odom,
                              rospy.Time.now(),
                              self.odom_frame_id,
                              self.world_frame_id)