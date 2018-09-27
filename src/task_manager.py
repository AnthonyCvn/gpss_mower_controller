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

# Python packages.
import numpy as np
from math import sqrt, atan2, pi
from scipy import interpolate, integrate
import matplotlib.pyplot as plt

# Specific controller's libraries.
from linear_mpc_controller import Controller
from toolbox import wraptopi

plotting = False


class TaskManager:
    """ Handle task execution of the robot.

    Receive a task and transmit the information to the controller:
        1) Extract the path from ExecuteTask message.
        2) Plan a trajectory for the mobile robot.
        3) Send the trajectory to the controller.
        4) Change the controller's status to active mode.

    Attributes:
        srv_execute_task    : ROS service to execute a task given by the motion planner.
        task_srv_name       : Name of the task execution service.
        robot_id            : Robot identification number.
        controller          : Controller object.
        Ts                  : Sampling time of the controller.
    """
    def __init__(self):
        self.srv_execute_task = None
        self.task_srv_name = "robot1/execute_task2"
        self.executer_command_topic = "robot1/controller/commands"
        self.executer_trajectories_topic = "robot1/controller/trajectories"
        self.controller_command = ControllerCommand.COMMAND_BRAKE

        self.robot_id = 1
        self.controller = Controller()
        self.Ts = 0.1
        self.desire_speed = 0.2
        self.a_tan_max = 0.5

    def run(self):
        """ Run the task manager by initializing a ROS service server. """
        rospy.loginfo("Robot #{0} is ready to receive a new task."
                      .format(self.robot_id))
        self.srv_execute_task = rospy.Service(self.task_srv_name, ExecuteTask, self.handle_task_execution)

        rospy.Subscriber(self.executer_command_topic, ControllerCommand, self.executer_command_cb)
        rospy.Subscriber(self.executer_trajectories_topic, ControllerTrajectoryChunkVec, self.executer_trajectories_cb)

    def handle_task_execution(self, task):
        """ Handle a new task for the controller.

        1) Extract the path from ExecuteTask message.
        2) Plan a trajectory for the mobile robot.
        3) Send the trajectory to the controller.
        4) Change the controller's status to active mode.

        """

        if len(task.task.path.path) <= 3:
            # Path way too short.
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            return 0

        self.controller.controller_report.robot_id = self.robot_id

        self.controller.sub_trajectories = self.trajectory_planning(task.task.path.path)
        self.controller.update_trajectory()
        rospy.loginfo("Robot #{0} run.".format(self.robot_id))

        if task.task.abort or self.controller.path_length == 0:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
        else:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE

        return 0

    def executer_command_cb(self, command):
        self.controller_command = command.command
        # Stop the car
        if self.controller_command == ControllerCommand.COMMAND_BRAKE:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to brake.".format(self.robot_id))

        # Set or change the active trajectory
        if self.controller_command == ControllerCommand.COMMAND_ACTIVATE:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to activate.".format(self.robot_id))

        # Set start time of tracking
        if self.controller_command == ControllerCommand.COMMAND_STARTTIME:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to start at {1} ns.".format(self.robot_id, command.start_time))

        # Recover after failure, ignored if there was no failure.
        if self.controller_command == ControllerCommand.COMMAND_RECOVER:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            rospy.loginfo("Robot #{0} receive order to recover.".format(self.robot_id))

    def executer_trajectories_cb(self, trajectories):
        while self.controller.controller_active:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_TERMINATE
            rospy.sleep(self.Ts)
            print"Stuck, Status = ", self.controller.controller_report.status

        print"N chunk : ", len(trajectories.chunks)
        if len(trajectories.chunks) < 2:
            rospy.loginfo("Robot #{0} receive a path way too short.".format(self.robot_id))
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            return

        path = []

        # Extract the path from the trajectory
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

        #if len(path) <= 3:
            # Path way too short.
            #self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_WAIT
            #rospy.loginfo("Path send to robot #{0} too short.".format(self.robot_id))
            #return 0

        self.controller.controller_report.robot_id = self.robot_id
        self.controller.sub_trajectories = self.trajectory_planning(path)
        self.controller.update_trajectory()

        if self.controller_command == ControllerCommand.COMMAND_STARTTIME:
            self.controller.controller_report.status = ControllerReport.CONTROLLER_STATUS_ACTIVE

    def trajectory_planning(self, path):
        """ Trajectory planner that assume a constant velocity between each points on the path.

        Args:
            path            : List of the robot's poses (x, y, phi) along the path.

        Returns:
            ref_trajectory  : Trajectory (x, y, phi, v, w) along the path (x, y, phi)

        """
        tol = 1e-12

        # Split the path with a different goal point at each change of direction
        goal_index = []
        goal_index.append(0)
        for i in range(len(path) - 2):
            dx_0 = path[i+1].pose.position.x - path[i].pose.position.x
            dx_1 = path[i+2].pose.position.x - path[i+1].pose.position.x
            dy_0 = path[i+1].pose.position.y - path[i].pose.position.y
            dy_1 = path[i+2].pose.position.y - path[i+1].pose.position.y
            yaw0 = atan2(dy_0, dx_0)
            dist_0 = np.linalg.norm([dx_0, dy_0])
            yaw1 = atan2(dy_1, dx_1)
            dist_1 = np.linalg.norm([dx_1, dy_1])
            if abs(wraptopi(yaw1 - yaw0)) > pi / 2 and dist_0 > tol and dist_1 > tol:
                goal_index.append(i + 1)
        goal_index.append(len(path) - 1)
        N_subpath = len(goal_index)-1

        # Separate the path for each goal
        sub_path = []
        for i in range(N_subpath):
            if len(path[goal_index[i]:goal_index[i+1]]) > 2:
                sub_path.append(path[goal_index[i]:goal_index[i+1]])

        # Check for each subpath if it is reversed
        is_subpath_reversed = []
        for s in range(N_subpath):
            yaw0 = atan2(sub_path[s][1].pose.position.y - sub_path[s][0].pose.position.y,
                         sub_path[s][1].pose.position.x - sub_path[s][0].pose.position.x)

            quaternion_subpath0 = (sub_path[s][1].pose.orientation.x,
                               sub_path[s][0].pose.orientation.y,
                               sub_path[s][0].pose.orientation.z,
                               sub_path[s][0].pose.orientation.w)
            euler_subpath0 = tf.transformations.euler_from_quaternion(quaternion_subpath0, axes='sxyz')
            phi0 = euler_subpath0[2]

            if abs(wraptopi(phi0 - yaw0)) > pi / 2:
                is_subpath_reversed.append(True)
            else:
                is_subpath_reversed.append(False)

        # Create the references trajectories
        ref_trajectory = []
        bspline = []
        for s in range(N_subpath):
            x = np.array([])
            y = np.array([])
            prev_x = sub_path[s][0].pose.position.x
            prev_y = sub_path[s][0].pose.position.y
            x = np.append(x, prev_x)
            y = np.append(y, prev_y)
            for i in range(1, len(sub_path[s])):
                new_x = sub_path[s][i].pose.position.x
                new_y = sub_path[s][i].pose.position.y
                distance = np.linalg.norm([new_x-prev_x, new_y-prev_y])
                if distance > tol:
                    x = np.append(x, new_x)
                    y = np.append(y, new_y)
                    prev_x = new_x
                    prev_y = new_y

            # Set the interpolation degree
            interp_degree = 5
            if len(x) <= interp_degree:
                interp_degree = int(len(x)-1)

            # Interpolate between points
            tck, t = interpolate.splprep([x, y], k=interp_degree, s=0.5) # s = 0.5 for smoothing

            bspline.append(tck)

            t = np.linspace(0, 1, num=2000, endpoint=True)
            xy = interpolate.splev(t, tck, der=0)
            dxydt = interpolate.splev(t, tck, der=1)

            arclength = integrate.cumtrapz(np.sqrt(dxydt[0] ** 2 + dxydt[1] ** 2), t, initial=0)

            # The segment length define the speed and the acceleration.
            speed_step = self.a_tan_max * self.Ts
            segment_speed = 0.05
            segment_length = segment_speed * self.Ts
            prev_length = 0.0
            t_equi = np.array([0])
            for l in zip(t, arclength):
                new_length = l[1] - prev_length
                if new_length >= segment_length:
                    t_equi = np.append(t_equi, l[0])
                    prev_length = l[1]
                    if segment_speed < self.desire_speed and l[0] <= 0.3:
                        segment_speed += speed_step
                        segment_length = segment_speed * self.Ts
                    if segment_speed > 0.2 and l[0] > 0.6:
                        segment_speed -= speed_step
                        segment_length = segment_speed * self.Ts

            t_equi = np.append(t_equi, 1)
            xy_equi = interpolate.splev(t_equi, tck, der=0)

            # Set the reference trajectory for each equidistant points
            ref_trajectory.append(np.zeros((len(t_equi), 5)))
            for i in range(len(t_equi)-1):
                # Evaluate BSpline object
                xy = interpolate.splev(t_equi[i], tck, der=0)
                dxydt = interpolate.splev(t_equi[i], tck, der=1)
                xy_next = interpolate.splev(t_equi[i+1], tck, der=0)
                dxydt_next = interpolate.splev(t_equi[i+1], tck, der=1)

                # Evaluate the distance between two points
                d = sqrt((xy_next[1] - xy[1]) ** 2 + (xy_next[0] - xy[0]) ** 2)

                # Store the pose at each point
                ref_trajectory[-1][i, 0] = xy[0]
                ref_trajectory[-1][i, 1] = xy[1]
                if is_subpath_reversed[s]:
                    # Backward direction
                    phi = wraptopi(np.arctan2(dxydt[1], dxydt[0])+np.pi)
                    phi_next = wraptopi(np.arctan2(dxydt_next[1], dxydt_next[0])+np.pi)
                    delta_phi = wraptopi(phi_next - phi)

                    ref_trajectory[-1][i, 2] = phi
                    ref_trajectory[-1][i, 3] = - d / self.Ts
                    ref_trajectory[-1][i, 4] = - delta_phi / self.Ts
                else:
                    # Frontward direction
                    phi = np.arctan2(dxydt[1], dxydt[0])
                    phi_next = np.arctan2(dxydt_next[1], dxydt_next[0])
                    delta_phi = wraptopi(phi_next - phi)

                    ref_trajectory[-1][i, 2] = phi
                    ref_trajectory[-1][i, 3] = d / self.Ts
                    ref_trajectory[-1][i, 4] = delta_phi / self.Ts

            # Compute the last point of the trajectory
            xy = interpolate.splev(t_equi[-1], tck, der=0)
            dxydt = interpolate.splev(t_equi[-1], tck, der=1)
            ref_trajectory[-1][-1, 0] = xy[0]
            ref_trajectory[-1][-1, 1] = xy[1]
            if is_subpath_reversed[s]:
                ref_trajectory[-1][-1, 2] = wraptopi(np.arctan2(dxydt[1], dxydt[0]) + np.pi)
            else:
                ref_trajectory[-1][-1, 2] = np.arctan2(dxydt[1], dxydt[0])
            ref_trajectory[-1][-1, 3] = 0.0
            ref_trajectory[-1][-1, 4] = 0.0

            # Plots
            if plotting:
                t = np.linspace(0, 1, num=2000, endpoint=True)
                xy = interpolate.splev(t, tck, der=0)
                dxydt = interpolate.splev(t, tck, der=1)

                plt.figure(2*s)
                plt.plot(x, y, 'bx', xy[0], xy[1], 'b', xy_equi[0], xy_equi[1], 'ro')
                plt.legend(['Received path', 'Interpolated B-spline', 'Equidistant points'], loc='best')
                plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])
                plt.title('B-Spline interpolation')

                if is_subpath_reversed[s]:
                    plt.figure(2*s+1)
                    plt.plot(t, wraptopi(np.arctan2(dxydt[1], dxydt[0])+np.pi) * 180 / np.pi, 'b')
                    plt.title('Phi')
                else:
                    plt.figure(2*s+1)
                    plt.plot(t, np.arctan2(dxydt[1], dxydt[0]) * 180 / np.pi, 'b')
                    plt.title('Phi')

        if plotting:
            plt.show()

        return ref_trajectory
