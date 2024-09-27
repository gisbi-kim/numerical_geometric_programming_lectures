"""
study mateiral recommendation:
    2016 ICRA SLAM tutorial 
    http://www.diag.uniroma1.it/~labrococo/tutorial_icra_2016/icra16_slam_tutorial_grisetti.pdf
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.pose import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_SE3_pose(T, color, label, ax):
    """Visualize SE(3) pose by plotting both the rotation and the translation."""
    origin = T[:3, 3]  # Translation part
    R = T[:3, :3]  # Rotation part

    for i in range(3):
        vec = R[:, i]
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=color[i],
            length=0.5,
            normalize=True,
            label=f"{label} Axis" if i == 0 else "",
        )
    ax.scatter(
        origin[0],
        origin[1],
        origin[2],
        color=color[0],
        s=50,
        label=f"{label} Translation",
    )


def create_rotation_matrix(roll, pitch, yaw, rotmatname="R"):

    #
    # change this values for various experiements
    init_bias_roll = np.deg2rad(180)
    init_bias_pitch = np.deg2rad(90)
    init_bias_yaw = np.deg2rad(10)
    #

    roll += init_bias_roll
    pitch += init_bias_pitch
    yaw += init_bias_yaw

    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    print(f"{rotmatname}: \n {R}")

    return R


def create_pose_matrix(rollpitchyaw, position, posename="T"):
    roll, pitch, yaw = rollpitchyaw
    tx, ty, tz = position

    # Initialize SE(3) matrix with rotation and translation
    R = create_rotation_matrix(roll, pitch, yaw, rotmatname=posename)
    t = np.array([tx, ty, tz]).reshape(3, 1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()

    print(f"{posename}: \n {T}")
    return T


# Function to average SE(3) poses (Rotation + Translation)
def average_SE3(poses, max_iters=300, tolerance=1e-6):
    # Initialize with identity (no rotation, no translation)
    T_current_solution = np.eye(4)

    for i in range(max_iters):
        H = np.zeros((6, 6))
        b = np.zeros(6)

        for T_i in poses:
            # Decompose rotation and translation
            R_i = T_i[:3, :3]
            t_i = T_i[:3, 3]

            e_i_rot = unskew(log_map(np.dot(T_current_solution[:3, :3].T, R_i)))
            e_i_trans = t_i - T_current_solution[:3, 3]
            e_i = np.hstack((e_i_rot, e_i_trans))

            J_i = -1 * np.eye(6)

            H += J_i.T @ J_i
            b += J_i.T @ e_i

        # Solve normal equation H * dtheta = -b
        dx = np.linalg.solve(H, -b)

        # Check for convergence
        if np.linalg.norm(dx) < tolerance:
            print(f"\nIteration terminated. The solution converged at iteration {i}\n")
            break

        # Update SE(3) pose
        T_current_solution = oplus(T_current_solution, dx)

        # Ensure the rotation part is a valid SO(3) matrix (det = 1)
        U, _, Vt = np.linalg.svd(T_current_solution[:3, :3])
        T_current_solution[:3, :3] = np.dot(U, Vt)
        if np.linalg.det(T_current_solution[:3, :3]) < 0:
            U[:, -1] *= -1
            T_current_solution[:3, :3] = np.dot(U, Vt)

    return T_current_solution


# Sample poses for SE(3) averaging
translation_noise_scale = 0.05

pose1_position = translation_noise_scale * np.array([1.0, 0.5, -0.2])
pose1_rollpitchyaw = [np.deg2rad(12.0), np.deg2rad(3.0), np.deg2rad(1.5)]

pose2_position = translation_noise_scale * np.array([-0.5, 1.2, 0.3])
pose2_rollpitchyaw = [np.deg2rad(-1.0), np.deg2rad(5.5), np.deg2rad(-2.0)]

pose3_position = translation_noise_scale * np.array([0.2, -0.8, 1.0])
pose3_rollpitchyaw = [np.deg2rad(3.5), np.deg2rad(-12.5), np.deg2rad(4.0)]

pose1 = create_pose_matrix(pose1_rollpitchyaw, pose1_position, "Pose1")
pose2 = create_pose_matrix(pose2_rollpitchyaw, pose2_position, "Pose2")
pose3 = create_pose_matrix(pose3_rollpitchyaw, pose3_position, "Pose3")

# Run SE(3) averaging
T_avg = average_SE3([pose1, pose2, pose3])

print("Optimized averaged pose (SE(3)):")
print(T_avg)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Frame axes
ax.quiver(0, 0, 0, 1, 0, 0, color="k", linewidth=1, label="Original X")
ax.quiver(0, 0, 0, 0, 1, 0, color="k", linewidth=1, label="Original Y")
ax.quiver(0, 0, 0, 0, 0, 1, color="k", linewidth=1, label="Original Z")

# Plot poses Pose1, Pose2, Pose3, and T_avg (averaged SE(3) pose)
plot_SE3_pose(pose1, ["r", "r", "r"], "Pose1", ax)
plot_SE3_pose(pose2, ["c", "c", "c"], "Pose2", ax)
plot_SE3_pose(pose3, ["orange", "orange", "orange"], "Pose3", ax)
plot_SE3_pose(T_avg, ["blue", "blue", "blue"], "T_avg", ax)

# Set axis limits and labels
ax.set_xlim([-0.4, 0.4])
ax.set_ylim([-0.4, 0.4])
ax.set_zlim([-0.4, 0.4])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Visualization of SE(3) Poses and Averaged Pose")
ax.legend()

plt.show()
