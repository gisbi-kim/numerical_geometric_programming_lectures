import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import copy


def print_optimized_delta_translation(pose_list):
    for idx, pose in enumerate(pose_list):
        if idx == len(pose_list) - 1:
            return

        pose_i = pose_list[idx]
        pose_j = pose_list[idx + 1]
        optimized_odom = pose_i.inverse() * pose_j

        # Rotation 객체로 변환
        rot = R.from_matrix(optimized_odom.R)
        roll, pitch, yaw = rot.as_euler("xyz", degrees=True)

        print(
            f"rel btn {idx} and {idx+1} [translation] is {optimized_odom.t[0]:.4f}, {optimized_odom.t[1]:.4f}, {optimized_odom.t[2]:.4f}"
        )
        print(
            f"rel btn {idx} and {idx+1} [roll, pitch, yaw] is {roll:.4f}, {pitch:.4f}, {yaw:.4f}"
        )


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
        # Add label only to the first node to avoid duplication
        if idx == 0:
            ax.scatter(
                origin[0],
                origin[1],
                origin[2],
                s=50,
                label=f"{label_prefix} {chr(ord('A') + idx)}",
                color=color,
            )
        else:
            ax.scatter(origin[0], origin[1], origin[2], s=50, color=color)
    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())


class Pose:
    def __init__(self, rotation=np.eye(3), translation=np.zeros(3)):
        self.R = rotation  # Rotation matrix
        self.t = translation  # Translation vector

    def as_matrix(self):
        """Return the homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @staticmethod
    def from_matrix(T):
        """Create a Pose from a homogeneous transformation matrix."""
        rotation = T[:3, :3]
        translation = T[:3, 3]
        return Pose(rotation, translation)

    def inverse(self):
        """Compute the inverse of this pose."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Pose(R_inv, t_inv)

    def noised(self):
        noise_rot = 0.1 * np.random.randn(3)
        noise_trans = 0.05 * np.random.randn(3)

        noised_R = self.R @ R.from_rotvec(noise_rot).as_matrix()
        noised_t = self.t + noise_trans

        return Pose(noised_R, noised_t)

    def __mul__(self, other):
        """Compose this pose with another pose."""
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return Pose(R_new, t_new)


def hat(v):
    """Skew-symmetric matrix for a vector."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vee(M):
    """Inverse of the hat operator."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def se3_to_SE3(xi):
    """Exponential map from se(3) to SE(3)."""
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        R = np.eye(3)
        t = v
    else:
        omega_hat = hat(omega / theta)
        R = (
            np.eye(3)
            + np.sin(theta) * omega_hat
            + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
        )
        V = (
            np.eye(3)
            + ((1 - np.cos(theta)) / theta) * omega_hat
            + ((theta - np.sin(theta)) / theta) * (omega_hat @ omega_hat)
        )
        t = V @ v
    return Pose(R, t)


def SE3_to_se3(T):
    """Logarithm map from SE(3) to se(3)."""
    R_mat = T.R
    t = T.t

    theta = np.arccos(np.clip((np.trace(R_mat) - 1) / 2, -1, 1))
    if theta < 1e-10:
        omega = np.zeros(3)
        v = t
    else:
        ln_R = (theta / (2 * np.sin(theta))) * (R_mat - R_mat.T)
        omega = vee(ln_R)
        omega_hat = ln_R
        A_inv = (
            np.eye(3)
            - 0.5 * omega_hat
            + ((1 / theta**2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta)))
            * (omega_hat @ omega_hat)
        )
        v = A_inv @ t
    xi = np.hstack((omega, v))
    return xi


def compute_between_error(T_i, T_j, Z_ij):
    """Compute the error for an edge."""
    # # E = T_i.inverse() * T_j  # Estimated relative transformation
    # # E_inv = Z_ij.inverse() * E

    Z_ij_est = T_i.inverse() * T_j

    # reason: Z_ij_est.oplus(e) = Z_ij
    # E = Z_ij_est.inverse() * Z_ij
    #   = (T_i.inverse() * T_j).inverse() * Z_ij
    #   = T_j.inverse() * T_i * Z_ij
    E = T_j.inverse() * T_i * Z_ij

    return SE3_to_se3(E)

    # """Compute the error for an edge."""
    # E = T_i.inverse() * T_j  # Estimated relative transformation
    # E_inv = Z_ij.inverse() * E

    # # Compute the rotational error using the rotation matrix logarithm
    # R_error = E_inv.R
    # rot_error = R.from_matrix(R_error).as_rotvec()  # Axis-angle representation

    # # Compute the translational error directly
    # t_error = E_inv.t

    # # Combine into a single error vector
    # e = np.hstack((rot_error, t_error))
    return e


def build_linear_system(poses, edges, priors=[], weight_prior=1e6):
    """Construct the linear system for optimization."""
    N = len(poses)
    H = np.zeros((6 * N, 6 * N))
    b = np.zeros(6 * N)
    for edge in edges:
        i, j, Z_ij = edge["i"], edge["j"], edge["measurement"]
        T_i = poses[i]
        T_j = poses[j]
        e_ij = compute_between_error(T_i, T_j, Z_ij)

        # Jacobians w.r.t T_i and T_j (approximate as identity for simplicity)
        # J_i = -np.eye(6)
        # J_j = np.eye(6)
        J_i = np.eye(6)
        J_j = -np.eye(6)

        # Generate arrays of indices instead of slices
        idx_i = np.arange(6 * i, 6 * i + 6)
        idx_j = np.arange(6 * j, 6 * j + 6)

        # Accumulate H and b using np.ix_ with arrays
        weight = edge.get("weight", 1.0)  # 가중치를 가져오고, 없으면 1로 기본 설정

        rot_weight = 0.1 * weight
        trans_weight = 0.5 * weight

        # 회전 부분 (앞 3개) 가중치 적용 (3x3 크기 맞추기)
        H[np.ix_(idx_i[:3], idx_i[:3])] += rot_weight * (J_i[:3, :3].T @ J_i[:3, :3])
        H[np.ix_(idx_i[:3], idx_j[:3])] += rot_weight * (J_i[:3, :3].T @ J_j[:3, :3])
        H[np.ix_(idx_j[:3], idx_i[:3])] += rot_weight * (J_j[:3, :3].T @ J_i[:3, :3])
        H[np.ix_(idx_j[:3], idx_j[:3])] += rot_weight * (J_j[:3, :3].T @ J_j[:3, :3])
        b[idx_i[:3]] += rot_weight * (J_i[:3, :3].T @ e_ij[:3])
        b[idx_j[:3]] += rot_weight * (J_j[:3, :3].T @ e_ij[:3])

        # 평행 이동 부분 (뒤 3개) 가중치 적용 (3x3 크기 맞추기)
        H[np.ix_(idx_i[3:], idx_i[3:])] += trans_weight * (J_i[3:, 3:].T @ J_i[3:, 3:])
        H[np.ix_(idx_i[3:], idx_j[3:])] += trans_weight * (J_i[3:, 3:].T @ J_j[3:, 3:])
        H[np.ix_(idx_j[3:], idx_i[3:])] += trans_weight * (J_j[3:, 3:].T @ J_i[3:, 3:])
        H[np.ix_(idx_j[3:], idx_j[3:])] += trans_weight * (J_j[3:, 3:].T @ J_j[3:, 3:])
        b[idx_i[3:]] += trans_weight * (J_i[3:, 3:].T @ e_ij[3:])
        b[idx_j[3:]] += trans_weight * (J_j[3:, 3:].T @ e_ij[3:])

        if 1 < abs(i - j):
            print(f"add loop closing factor btn {i} and {j}!")

        # 원소가 있는 부분을 1, 없는 부분을 0으로 변환
        draw_jacobian_mat = 0
        if draw_jacobian_mat:
            binary_matrix = (H != 0).astype(int)
            plt.imshow(binary_matrix, cmap="gray")  # 흑백으로 시각화
            plt.colorbar(label="0: Empty, 1: Not Empty")
            plt.show()

    # Process prior residuals
    for prior in priors:
        i = prior["i"]
        Z_i = prior["measurement"]
        T_i = poses[i]

        e_i = SE3_to_se3(T_i.inverse() * Z_i)
        J_i = -np.eye(6)

        idx_i = np.arange(6 * i, 6 * i + 6)
        H[np.ix_(idx_i, idx_i)] += weight_prior * (J_i.T @ J_i)
        b[idx_i] += weight_prior * (J_i.T @ e_i)
    return H, b


def pose_graph_optimization(poses, edges, iterations=30):
    """Perform pose-graph optimization."""
    N = len(poses)

    # priors = [{'i': 0, 'measurement': Pose()}, {'i': N-1, 'measurement': initial_poses[-1]}]
    # priors = [{"i": 0, "measurement": Pose()}]
    priors = [{"i": N - 1, "measurement": Pose()}]
    # priors = [{'i': int(N/2) - 1, 'measurement': Pose()}]

    print(f"priors: {priors}")
    weight_prior = 1e8
    for iter in range(iterations):
        print(f"iter {iter}")
        H, b = build_linear_system(poses, edges, priors, weight_prior)
        delta = np.linalg.solve(H, -b)
        # Update poses
        for i in range(N):
            idx = slice(6 * i, 6 * i + 6)
            xi = delta[idx]
            delta_pose = se3_to_SE3(xi)

            # poses[i] = delta_pose * poses[i]
            poses[i] = poses[i] * delta_pose
    return poses


# Example usage:
# Initialize poses and edges

num_nodes = 50

odom_rot_yaw_deg = 330.0 / num_nodes
odom_rot_yaw_rad = np.deg2rad(odom_rot_yaw_deg)

r = 3.0  # 예: 반지름 1.0
move_forward_size = 2 * r * np.sin(np.deg2rad(180.0 / num_nodes))

move_once_pose = Pose(
    R.from_euler("z", odom_rot_yaw_rad).as_matrix()[:3, :3],
    np.array([move_forward_size, 0, 0]),
)

between_odom_weight = 1.0
between_loop_weight = 1.0

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

# Create initial poses without noise
initial_poses = []
initial_poses.append(Pose())  # Pose 0 at identity
for i in range(1, num_nodes):
    prev_pose = initial_poses[i - 1]
    measurement = move_once_pose
    new_pose = prev_pose * measurement  # No noise added
    print(f"new_pose \n{new_pose.as_matrix()}")
    initial_poses.append(new_pose)

# Now, add edges between nodes that are k steps apart
for dense_connection_k in []:
    for i in range(num_nodes - dense_connection_k):
        # Compute the measurement between node i and node i+k
        Z_ik = initial_poses[i].inverse() * initial_poses[i + dense_connection_k]
        # Optionally, add noise to Z_ik
        # Z_ik = Z_ik.noised()
        edges.append({"i": i, "j": i + dense_connection_k, "measurement": Z_ik})

# loop closing
if 1:
    edges.append(
        {
            "i": 0,
            "j": num_nodes - 1,
            "measurement": Pose(
                R.from_euler("z", np.deg2rad(0)).as_matrix()[:3, :3],
                np.array([0, 0, 0]),
            ),
            "weight": between_loop_weight,
        }
    )

# Run optimization
optimized_poses = pose_graph_optimization(initial_poses.copy(), edges)

print_optimized_delta_translation(optimized_poses)
print(f"\n when move_forward_size {move_forward_size}")

#################################################
max_ax_lim = 10
# Plot initial poses
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_poses(initial_poses, ax, color="b", label_prefix="Initial Pose")
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-max_ax_lim, max_ax_lim])
ax.set_ylim([-max_ax_lim, max_ax_lim])
ax.set_zlim([-max_ax_lim, max_ax_lim])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Before Optimization")
plt.show()

# Plot optimized poses
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_poses(optimized_poses, ax, color="g", label_prefix="Optimized Pose")
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-max_ax_lim, max_ax_lim])
ax.set_ylim([-max_ax_lim, max_ax_lim])
ax.set_zlim([-max_ax_lim, max_ax_lim])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("After Optimization")
plt.show()
