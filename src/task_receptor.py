#!/usr/bin/env python

# ROS libraries for Python.
import rospy
import tf

# ROS messages.
from orunav_msgs.srv import ExecuteTask
from orunav_msgs.msg import ControllerReport
from orunav_msgs.msg import ControllerTrajectoryChunkVec
from orunav_msgs.msg import ControllerCommand
from geometry_msgs.msg import PoseStamped

# Specific controller's libraries.
from trajectory_builder import get_trajectory_from_path


class TaskReceptor:
    """ Handle task execution.

    Receive a task and transmit the information to the regulator:
        1) Extract the path from the task message.
        2) Build a reference trajectory from the received path.
        3) Send the trajectory to the regulator.
        4) Change the regulator's status.

    Args:
        robot_id                : Robot identification number.
        regulator               : Regulator that calculate the input command for the plant.
        desire_speed            : Desire forward speed of the mobile robot in m/s.

    Attributes:
        robot_id                : Robot identification number.
        controller              : Regulator that calculate the input command for the plant.
        desire_speed            : Desire forward speed of the mobile robot in m/s.
        Ts                      : Sampling time of the controller.
        a_tan_max               : Maximum tangential acceleration of the robot.
        controller_command      : Store the received command.
        srv_execute_task        : ROS service to execute a task given by the motion planner.

    Note:
        The task is receive either by ExecuteTask service or by commands and trajectories topics.
    """
    def __init__(self, robot_id, regulator, desire_speed, deceleration_distance):
        self.robot_id = robot_id
        self.controller = regulator
        self.desire_speed = desire_speed
        self.Ts = regulator.Ts
        self.a_tan_max = regulator.a_tan_max
        self.deceleration_distance = deceleration_distance
        self.controller_command = ControllerCommand.COMMAND_BRAKE

        # Run the task manager by initializing a ROS service server.
        rospy.loginfo("Robot #{0} is ready to receive a new task.".format(self.robot_id))

        task_srv_name = "robot{0}/execute_task2".format(robot_id)
        self.srv_execute_task = rospy.Service(task_srv_name, ExecuteTask, self.task_execution)

        execute_command_topic = "robot{0}/controller/commands".format(robot_id)
        rospy.Subscriber(execute_command_topic, ControllerCommand, self.execute_command_cb)

        execute_trajectories_topic = "robot{0}/controller/trajectories".format(robot_id)
        rospy.Subscriber(execute_trajectories_topic, ControllerTrajectoryChunkVec, self.execute_trajectories_cb)

    def task_execution(self, task):
        """ Handle a new task for the controller.

        1) Extract the path from the task message.
        2) Build a reference trajectory from the received path.
        3) Send the trajectory to the regulator.
        4) Change the regulator's status.

        """

        # 1) Extract the path Task message.
        path = task.task.path.path
        if len(path) <= 3:
            # Path way too short.
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            return 0

        # 2) Build a reference trajectory from the received path.
        self.controller.sub_trajectories = get_trajectory_from_path(path, self.Ts, self.desire_speed,
                                                                    self.a_tan_max, self.deceleration_distance)

        # 3) Send the trajectory to the regulator.
        self.controller.update_trajectory()
        rospy.loginfo("Robot #{0} run.".format(self.robot_id))

        # 4) Change the regulator status to active mode.
        if task.task.abort or self.controller.path_length == 0:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
        else:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE

        return 0

    def execute_command_cb(self, command):
        """ Callback that change the regulator's status. """

        self.controller_command = command.command
        # Stop the car
        if self.controller_command == ControllerCommand.COMMAND_BRAKE:
            #self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to brake.".format(self.robot_id))

        # Set or change the active trajectory
        if self.controller_command == ControllerCommand.COMMAND_ACTIVATE:
            #self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to activate.".format(self.robot_id))

        # Set start time of tracking
        if self.controller_command == ControllerCommand.COMMAND_STARTTIME:
            #self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to start.")

        # Recover after failure, ignored if there was no failure.
        if self.controller_command == ControllerCommand.COMMAND_RECOVER:
            #self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to recover.".format(self.robot_id))

    def execute_trajectories_cb(self, trajectories):
        """ Handle a new task for the controller.

        0) Set terminate mode if the controller is active.
        1) Extract the path from the received trajectory.
        2) Build a reference trajectory from the extracted path.
        3) Send the trajectory to the regulator.
        4) Change the regulator's status.

        """
        rospy.loginfo("Robot #{0} received a new path.".format(self.robot_id))

        # 0) Set terminate mode if the controller is active.
        if self.controller.controller_active:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_TERMINATE
            rospy.sleep(2*self.Ts)

        # Reference path and trajectory
        self.controller.sub_trajectories = None
        self.controller.n_subgoal = 0
        self.controller.ref_path = None
        self.controller.ref_trajectory = None
        self.controller.path_length = 0

        # Current reference index and trajectory
        self.controller.current_trajectory = None
        self.controller.index_path = 0
        self.controller.distance_to_path = None
        self.controller.max_distance_to_path = 1.0
        self.controller.index_ahead = 10
        self.controller.index_back = 10
        self.controller.final_index_counter = 0


        # 1) Extract the path from the received trajectory.
        path = []

        if len(trajectories.chunks) < 2:
            rospy.loginfo("Robot #{0} received a path way too short.".format(self.robot_id))
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            return

        for i in range(len(trajectories.chunks)):
            for j in range(len(trajectories.chunks[i].steps)):
                pose = PoseStamped()
                yaw = trajectories.chunks[i].steps[j].state.orientation_angle
                quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                pose.pose.position.x = trajectories.chunks[i].steps[j].state.position_x
                pose.pose.position.y = trajectories.chunks[i].steps[j].state.position_y
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]
                path.append(pose)

        # 2) Build a reference trajectory from the extracted path.
        self.controller.sub_trajectories = get_trajectory_from_path(path, self.Ts, self.desire_speed,
                                                                    self.a_tan_max, self.deceleration_distance)

        # 3) Send the trajectory to the regulator.
        self.controller.update_trajectory()

        # 4) Change the regulator status.
        #if self.controller_command == ControllerCommand.COMMAND_STARTTIME:
        self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE
