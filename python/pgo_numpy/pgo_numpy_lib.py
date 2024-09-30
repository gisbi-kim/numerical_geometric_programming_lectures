import numpy as np
import matplotlib.pyplot as plt


def plot_initial_poses(plot_poses, initial_poses, max_ax_lim):
    """Plot the initial poses."""
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


def plot_optimized_poses(plot_poses, optimized_poses, max_ax_lim):
    """Plot the optimized poses."""
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


def compute_between_error(T_i, T_j, Z_ij_measurement):
    """
    The observation model is
       Z_ij_est.oplus(e) = Z_ij_measurement
       , where Z_ij_est = T_i.inverse() * T_j
    Thus,
       E = Z_ij_est.inverse() * Z_ij
         = (T_i.inverse() * T_j).inverse() * Z_ij
         = (T_j.inverse() * T_i) * Z_ij
    """
    E = (T_j.inverse() * T_i) * Z_ij_measurement
    return SE3_to_se3(E)


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

        J_i = np.eye(6)
        J_j = -np.eye(6)

        idx_i = np.arange(6 * i, 6 * i + 6)
        idx_j = np.arange(6 * j, 6 * j + 6)

        weight = edge.get("weight", 1.0)

        rot_weight = 1.0 * weight
        trans_weight = 1.0 * weight

        H[np.ix_(idx_i[:3], idx_i[:3])] += rot_weight * (J_i[:3, :3].T @ J_i[:3, :3])
        H[np.ix_(idx_i[:3], idx_j[:3])] += rot_weight * (J_i[:3, :3].T @ J_j[:3, :3])
        H[np.ix_(idx_j[:3], idx_i[:3])] += rot_weight * (J_j[:3, :3].T @ J_i[:3, :3])
        H[np.ix_(idx_j[:3], idx_j[:3])] += rot_weight * (J_j[:3, :3].T @ J_j[:3, :3])
        b[idx_i[:3]] += rot_weight * (J_i[:3, :3].T @ e_ij[:3])
        b[idx_j[:3]] += rot_weight * (J_j[:3, :3].T @ e_ij[:3])

        H[np.ix_(idx_i[3:], idx_i[3:])] += trans_weight * (J_i[3:, 3:].T @ J_i[3:, 3:])
        H[np.ix_(idx_i[3:], idx_j[3:])] += trans_weight * (J_i[3:, 3:].T @ J_j[3:, 3:])
        H[np.ix_(idx_j[3:], idx_i[3:])] += trans_weight * (J_j[3:, 3:].T @ J_i[3:, 3:])
        H[np.ix_(idx_j[3:], idx_j[3:])] += trans_weight * (J_j[3:, 3:].T @ J_j[3:, 3:])
        b[idx_i[3:]] += trans_weight * (J_i[3:, 3:].T @ e_ij[3:])
        b[idx_j[3:]] += trans_weight * (J_j[3:, 3:].T @ e_ij[3:])

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

    priors = [{"i": 0, "measurement": Pose()}]

    weight_prior = 1e8
    for iter in range(iterations):
        H, b = build_linear_system(poses, edges, priors, weight_prior)
        delta = np.linalg.solve(H, -b)
        for i in range(N):
            idx = slice(6 * i, 6 * i + 6)
            xi = delta[idx]
            delta_pose = se3_to_SE3(xi)

            poses[i] = poses[i] * delta_pose
    return poses
