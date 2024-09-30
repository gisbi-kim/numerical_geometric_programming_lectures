import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gtsam
from gtsam import (
    NonlinearFactorGraph,
    Values,
    Pose3,
    Rot3,
    Point3,
    PriorFactorPose3,
    BetweenFactorPose3,
    noiseModel,
)
from math import radians


def create_pose(roll, pitch, yaw, x, y, z):
    """
    Create a Pose3 object from roll, pitch, yaw (in radians) and position coordinates.
    """
    rotation = Rot3.RzRyRx(roll, pitch, yaw)
    translation = Point3(x, y, z)
    return Pose3(rotation, translation)


def add_noise_to_pose(pose, rot_noise_std=0.01, trans_noise_std=0.01):
    """
    Add Gaussian noise to a Pose3 object.
    """
    # Add noise to the rotation
    noise_rot = rot_noise_std * np.random.randn(3)  # Rotation vector noise
    noisy_rot = pose.rotation().compose(
        Rot3.Rx(noise_rot[0])
        .compose(Rot3.Ry(noise_rot[1]))
        .compose(Rot3.Rz(noise_rot[2]))
    )

    # Add noise to the translation
    noise_trans = trans_noise_std * np.random.randn(3)
    noisy_trans = Point3(
        pose.x() + noise_trans[0], pose.y() + noise_trans[1], pose.z() + noise_trans[2]
    )

    return Pose3(noisy_rot, noisy_trans)


def plot_poses(pose_list, ax, color="b", label_prefix="Pose"):
    """
    Visualize a list of Pose3 objects on a 3D plot.
    """
    for idx, pose in enumerate(pose_list):
        origin = np.array([pose.x(), pose.y(), pose.z()])
        R = pose.rotation().matrix()
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


def plot_before_optimization(initial_estimates, ax_lim=3.0):
    """
    Plot poses before optimization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(initial_estimates, ax, color="b", label_prefix="Initial Pose")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_zlim([-ax_lim, ax_lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Before Optimization")
    plt.show()


def plot_after_optimization(optimized_poses, ax_lim=3.0):
    """
    Plot poses after optimization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(optimized_poses, ax, color="g", label_prefix="Optimized Pose")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_zlim([-ax_lim, ax_lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("After Optimization")
    plt.show()


def main():
    # Initialize factor graph and initial estimates
    graph = NonlinearFactorGraph()
    initial = Values()

    # Define noise models
    prior_noise = noiseModel.Diagonal.Sigmas(
        np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    )
    odometry_noise = noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    )
    loop_noise = noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    )

    # Define relative motion: move forward by 1 unit in x and rotate 90 degrees around the z-axis
    delta_x = 1.0
    delta_yaw = np.deg2rad(80)
    move_once = create_pose(0.0, 0.0, delta_yaw, delta_x, 0.0, 0.0)

    # Define node keys (using integer keys)
    A, B, C, D, E = range(5)

    # Create poses from A to E
    poses = [Pose3()]
    for i in range(1, 5):
        prev_pose = poses[i - 1]
        poses.append(prev_pose.compose(move_once))

    # Add a prior factor at node A
    graph.add(PriorFactorPose3(A, poses[0], prior_noise))

    # Add between factors (A-B, B-C, C-D, D-E)
    for i in range(4):
        graph.add(BetweenFactorPose3(i, i + 1, move_once, odometry_noise))

    # Add loop closure factor (A-E)
    graph.add(BetweenFactorPose3(A, E, Pose3(), loop_noise))

    # Add noisy initial estimates
    initial_estimates = []
    for i, key in enumerate([A, B, C, D, E]):
        noisy_pose = add_noise_to_pose(poses[i])
        initial.insert(key, noisy_pose)
        initial_estimates.append(noisy_pose)

    # Plot before optimization
    plot_before_optimization(initial_estimates)

    # Perform graph optimization
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # Extract optimized poses
    optimized_poses = []
    for key in [A, B, C, D, E]:
        optimized_poses.append(result.atPose3(key))

    # Plot after optimization
    plot_after_optimization(optimized_poses)

    # Print optimized poses
    print("\nOptimized Poses:")
    for i, key in enumerate([A, B, C, D, E]):
        pose = result.atPose3(key)
        print(f"Node {chr(ord('A') + i)}:\n{pose}\n")


if __name__ == "__main__":
    main()
