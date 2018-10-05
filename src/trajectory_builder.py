#!/usr/bin/env python

# ROS libraries for Python.
import tf

# Python packages.
import numpy as np
from math import atan2, pi
from scipy import interpolate, integrate
import matplotlib.pyplot as plt

# Specific controller's libraries.
from toolbox import wraptopi


def get_trajectory_from_path(path, Ts, desire_speed, a_tan_max, deceleration_distance, plot=False):
    """ Trajectory planner that assume a constant velocity between each points on the path.

    Args:
        path                    : List of the robot's poses (x, y, phi) along the path.
        Ts                      : Sampling time of the regulator.
        desire_speed            : Desire speed along the given path.
        a_tan_max               : Maximal tangential acceleration.
        deceleration_distance   : Distance in meters to decelerate.
        plot                    : Plot the result if True. (Blocking until closing the windows)

    Returns:
        ref_trajectory  : Trajectory (x, y, phi, v, w) along the path (x, y, phi)

    """
    tol = 1e-6

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
        tck, t = interpolate.splprep([x, y], k=interp_degree, s=0.02) # s = 0.5 for smoothing

        bspline.append(tck)

        t = np.linspace(0, 1, num=2000, endpoint=True)
        dxy = interpolate.splev(t, tck, der=1)

        arclength = integrate.cumtrapz(np.sqrt(dxy[0] ** 2 + dxy[1] ** 2), t, initial=0)
        path_length = arclength[-1]

        # The segment length define the speed and the acceleration.
        speed_step_forward = a_tan_max * Ts
        n_deceleration_step = deceleration_distance / (desire_speed * Ts)
        speed_step_backward = desire_speed / n_deceleration_step

        minimum_speed = 0.15

        segment_speed = minimum_speed
        segment_length = segment_speed * Ts
        prev_length = 0.0
        t_equi = np.array([0])
        s_speed = np.array([])
        for l in zip(t, arclength):
            new_length = l[1] - prev_length
            if new_length >= segment_length:
                t_equi = np.append(t_equi, l[0])
                s_speed = np.append(s_speed, segment_speed)
                prev_length = l[1]
                if segment_speed > minimum_speed and (path_length-l[1]) < deceleration_distance:
                    segment_speed -= speed_step_backward
                    segment_length = segment_speed * Ts
                elif segment_speed < desire_speed:
                    segment_speed += speed_step_forward
                    if segment_speed > desire_speed:
                        segment_speed = desire_speed
                    segment_length = segment_speed * Ts

        s_speed = np.append(s_speed, 0.0)

        t_equi = np.append(t_equi, 1)
        xy_equi = interpolate.splev(t_equi, tck, der=0)

        # Set the reference trajectory for each equidistant points
        ref_trajectory.append(np.zeros((len(t_equi), 5)))
        for i in range(len(t_equi)-1):
            # Evaluate BSpline object
            xy = interpolate.splev(t_equi[i], tck, der=0)
            dxy = interpolate.splev(t_equi[i], tck, der=1)

            if interp_degree > 1:
                d2xy = interpolate.splev(t_equi[i], tck, der=2)
                # Path curvature
                kappa = (dxy[0] * d2xy[1] - dxy[1] * d2xy[0]) / (dxy[0]**2 + dxy[1]**2)**(3.0/2.0)
            else:
                kappa = 0.0

            # Segment reference speed
            v = s_speed[i]

            # Store the pose at each point
            ref_trajectory[-1][i, 0] = xy[0]
            ref_trajectory[-1][i, 1] = xy[1]
            if is_subpath_reversed[s]:
                # Backward direction
                phi = wraptopi(np.arctan2(dxy[1], dxy[0])+np.pi)
                ref_trajectory[-1][i, 2] = phi
                ref_trajectory[-1][i, 3] = - v
                ref_trajectory[-1][i, 4] = - kappa * v
            else:
                # Frontward direction
                phi = np.arctan2(dxy[1], dxy[0])
                ref_trajectory[-1][i, 2] = phi
                ref_trajectory[-1][i, 3] = v
                ref_trajectory[-1][i, 4] = kappa * v

        # Compute the last point of the trajectory
        xy = interpolate.splev(t_equi[-1], tck, der=0)
        dxy = interpolate.splev(t_equi[-1], tck, der=1)
        ref_trajectory[-1][-1, 0] = xy[0]
        ref_trajectory[-1][-1, 1] = xy[1]
        if is_subpath_reversed[s]:
            ref_trajectory[-1][-1, 2] = wraptopi(np.arctan2(dxy[1], dxy[0]) + np.pi)
        else:
            ref_trajectory[-1][-1, 2] = np.arctan2(dxy[1], dxy[0])
        ref_trajectory[-1][-1, 3] = 0.0
        ref_trajectory[-1][-1, 4] = 0.0

        # Plots
        if plot:
            t = np.linspace(0, 1, num=2000, endpoint=True)
            xy = interpolate.splev(t, tck, der=0)
            dxy = interpolate.splev(t, tck, der=1)

            plt.figure(2*s)
            plt.plot(x, y, 'bx', xy[0], xy[1], 'b', xy_equi[0], xy_equi[1], 'ro')
            plt.legend(['Received path', 'Interpolated B-spline', 'Equidistant points'], loc='best')
            plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])
            plt.title('B-Spline interpolation')

            if is_subpath_reversed[s]:
                plt.figure(2*s+1)
                plt.plot(t, wraptopi(np.arctan2(dxy[1], dxy[0])+np.pi) * 180 / np.pi, 'b')
                plt.title('Phi')
            else:
                plt.figure(2*s+1)
                plt.plot(t, np.arctan2(dxy[1], dxy[0]) * 180 / np.pi, 'b')
                plt.title('Phi')

    if plot:
        plt.show()

    return ref_trajectory
