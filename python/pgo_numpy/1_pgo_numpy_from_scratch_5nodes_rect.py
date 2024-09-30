import numpy as np
import matplotlib.pyplot as plt

from pgo_numpy_lib import *


def plot_poses(pose_list, ax, color="b", label_prefix="Pose"):
    """
    Visualize a list of Pose objects in a 3D plot.
    """
    for idx, pose in enumerate(pose_list):
        origin = pose.t  # Translation vector
        R = pose.R  # Rotation matrix
        colors = ["r", "g", "b"]
        for i in range(3):
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                R[0, i],
                R[1, i],
                R[2, i],
                color=colors[i],
                length=0.3,
                normalize=True,
            )

        ax.scatter(
            origin[0],
            origin[1],
            origin[2],
            s=50,
            label=f"{label_prefix} {chr(ord('A') + idx)}",
        )

    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())


# Example usage:
def main():

    # Initialize poses and edges
    num_nodes = 5

    odom_rot_yaw_deg = 340.0 / (num_nodes - 1)  # 85 deg per odometry move one step
    odom_rot_yaw_rad = np.deg2rad(odom_rot_yaw_deg)

    r = 2.0  # Example: radius 1.0
    move_forward_size = 2 * r * np.sin(np.deg2rad(180.0 / num_nodes))

    move_once_pose = Pose(
        # yaw (on SE(2) plane) rotation
        np.array(
            [
                [np.cos(odom_rot_yaw_rad), -np.sin(odom_rot_yaw_rad), 0],
                [np.sin(odom_rot_yaw_rad), np.cos(odom_rot_yaw_rad), 0],
                [0, 0, 1],
            ]
        ),
        # go forward (x axis)
        np.array([move_forward_size, 0, 0]),
    )

    between_odom_weight = 1.0
    between_loop_weight = 100.0

    edges = []
    for node_ii in range(num_nodes - 1):
        edges.append(
            {
                "i": node_ii,
                "j": node_ii + 1,
                "measurement": move_once_pose,
                "weight": between_odom_weight,
            }
        )

    initial_poses = []
    initial_poses.append(Pose())  # Pose 0 at identity
    for i in range(1, num_nodes):
        prev_pose = initial_poses[i - 1]
        measurement = move_once_pose
        new_pose = prev_pose * measurement
        initial_poses.append(new_pose)

    # loop closing
    same_place_relative_pose = Pose()
    edges.append(
        {
            "i": 0,
            "j": num_nodes - 1,
            "measurement": same_place_relative_pose,
            "weight": between_loop_weight,
        }
    )

    # Run optimization
    optimized_poses = pose_graph_optimization(initial_poses.copy(), edges)

    # Set axis limits
    max_ax_lim = 4.0
    plot_initial_poses(plot_poses, initial_poses, max_ax_lim)
    plot_optimized_poses(plot_poses, optimized_poses, max_ax_lim)


if __name__ == "__main__":
    main()
