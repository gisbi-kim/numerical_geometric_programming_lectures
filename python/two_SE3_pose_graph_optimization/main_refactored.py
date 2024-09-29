import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.pose_module import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_rotation_matrix(roll, pitch, yaw):
    """
    롤, 피치, 요 각을 사용하여 회전 행렬을 생성합니다.
    """
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
    R = R_z @ R_y @ R_x
    return R


def create_pose_matrix(roll, pitch, yaw, position):
    """
    롤, 피치, 요 각과 위치 벡터를 사용하여 SE(3) 행렬을 생성합니다.
    """
    R = create_rotation_matrix(roll, pitch, yaw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


class Node:
    """
    포즈 그래프의 노드를 나타내는 클래스입니다.
    각 노드는 고유한 문자열 ID와 SE(3) 포즈를 가집니다.
    """

    def __init__(self, node_id, initial_pose=None):
        self.node_id = node_id
        if initial_pose is None:
            self.pose = np.eye(4)  # 기본 포즈는 항등 행렬
        else:
            self.pose = initial_pose.copy()


class Factor:
    """
    포즈 그래프의 팩터를 나타내는 기본 클래스입니다.
    """

    pass


class PriorFactor(Factor):
    """
    노드에 대한 Prior 제약을 나타내는 클래스입니다.
    """

    def __init__(self, node_id, prior_pose):
        self.node_id = node_id
        self.prior_pose = prior_pose.copy()


class BetweenFactor(Factor):
    """
    두 노드 간의 Between Residual을 나타내는 클래스입니다.
    """

    def __init__(self, from_node_id, to_node_id, relative_pose):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.relative_pose = relative_pose.copy()


class PoseGraph:
    """
    포즈 그래프 최적화를 수행하는 클래스입니다.
    """

    def __init__(self):
        self.nodes = {}  # node_id: Node
        self.prior_factors = []  # List of PriorFactor
        self.between_factors = []  # List of BetweenFactor

    def generate_node_at_graph(self, node_id, initial_pose=None):
        """
        그래프에 노드를 추가합니다.

        Parameters:
        - node_id: 노드의 고유 문자열 ID
        - initial_pose: 초기 SE(3) 포즈 (옵션)
        """
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self.nodes[node_id] = Node(node_id, initial_pose)

    def add_prior_factor(self, node_id, prior_pose):
        """
        노드에 대한 Prior Factor를 추가합니다.

        Parameters:
        - node_id: 노드의 문자열 ID
        - prior_pose: Prior로 설정할 SE(3) 포즈
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist.")
        self.prior_factors.append(PriorFactor(node_id, prior_pose))

    def add_between_factor(self, from_node_id, to_node_id, relative_pose):
        """
        두 노드 간의 Between Factor를 추가합니다.

        Parameters:
        - from_node_id: 시작 노드의 문자열 ID
        - to_node_id: 도착 노드의 문자열 ID
        - relative_pose: from_node_id에서 to_node_id로의 상대 SE(3) 포즈
        """
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError(
                f"Both nodes '{from_node_id}' and '{to_node_id}' must exist."
            )
        self.between_factors.append(
            BetweenFactor(from_node_id, to_node_id, relative_pose)
        )

    def solve_graph(self, max_iterations=100, tolerance=1e-6):
        """
        포즈 그래프를 최적화합니다.

        Parameters:
        - max_iterations: 최적화의 최대 반복 횟수
        - tolerance: 수렴을 판단할 허용 오차
        """
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        num_nodes = len(node_ids)
        H = np.zeros((6 * num_nodes, 6 * num_nodes))
        b = np.zeros(6 * num_nodes)

        for iteration in range(max_iterations):
            H.fill(0)
            b.fill(0)

            # Prior Residuals
            for prior in self.prior_factors:
                i = node_id_to_idx[prior.node_id]
                T_est = self.nodes[prior.node_id].pose
                T_prior = prior.prior_pose

                # Residual 계산: log_map(inv(T_prior) * T_est)
                T_residual = np.linalg.inv(T_prior) @ T_est
                R_residual = T_residual[:3, :3]
                dtheta = Log_map(R_residual)
                dt = T_residual[:3, 3]

                residual = np.hstack((dtheta, dt))

                # Jacobian은 단위 행렬 (Prior Residual은 노드 자체에만 영향)
                J = np.eye(6)

                # Hessian 및 b 업데이트
                H_i = J.T @ J
                b_i = J.T @ residual

                H[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] += H_i
                b[6 * i : 6 * (i + 1)] += b_i

            # Between Residuals
            for between in self.between_factors:
                i = node_id_to_idx[between.from_node_id]
                j = node_id_to_idx[between.to_node_id]
                T_i = self.nodes[between.from_node_id].pose
                T_j = self.nodes[between.to_node_id].pose
                T_AB = between.relative_pose

                # Residual 계산: log_map(inv(T_i) * T_j * inv(T_AB))
                T_residual = np.linalg.inv(T_i) @ T_j @ np.linalg.inv(T_AB)
                R_residual = T_residual[:3, :3]
                dtheta = Log_map(R_residual)
                dt = T_residual[:3, 3]
                residual = np.hstack((dtheta, dt))

                # Jacobian 설정: 노드 i는 -I, 노드 j는 +I
                J_i = -np.eye(6)
                J_j = np.eye(6)

                # Weighting: Between Residual의 가중치 0.5
                weight_between = 0.5
                residual_weighted = residual * np.sqrt(weight_between)
                J_i_weighted = J_i * np.sqrt(weight_between)
                J_j_weighted = J_j * np.sqrt(weight_between)

                # Hessian 및 b 업데이트
                H_i = J_i_weighted.T @ J_i_weighted
                H_j = J_j_weighted.T @ J_j_weighted
                H_ij = J_i_weighted.T @ J_j_weighted

                H[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] += H_i
                H[6 * j : 6 * (j + 1), 6 * j : 6 * (j + 1)] += H_j
                H[6 * i : 6 * (i + 1), 6 * j : 6 * (j + 1)] += H_ij
                H[6 * j : 6 * (j + 1), 6 * i : 6 * (i + 1)] += H_ij.T

                b[6 * i : 6 * (i + 1)] += J_i_weighted.T @ residual_weighted
                b[6 * j : 6 * (j + 1)] += J_j_weighted.T @ residual_weighted

            # Solve H * dx = -b
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                print("Singular matrix encountered during optimization.")
                break

            # Check convergence
            if np.linalg.norm(dx) < tolerance:
                print(f"Converged at iteration {iteration}")
                break

            # Update poses
            for node_id in node_ids:
                idx = node_id_to_idx[node_id]
                tangent_vec = dx[6 * idx : 6 * (idx + 1)]
                self.nodes[node_id].pose = oplus(self.nodes[node_id].pose, tangent_vec)

            if iteration == max_iterations - 1:
                print("Reached maximum iterations without convergence.")

    def get_solution(self, node_id):
        """
        최적화된 포즈를 반환합니다.

        Parameters:
        - node_id: 노드의 문자열 ID

        Returns:
        - SE(3) 포즈 행렬
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist.")
        return self.nodes[node_id].pose.copy()


def plot_graph(pose_graph, ax, color="b", label_prefix="Pose"):
    """
    포즈 그래프의 노드 포즈를 3D 플롯에 시각화합니다.

    Parameters:
    - pose_graph: PoseGraph 객체
    - ax: matplotlib 3D Axes 객체
    - color: 화살표 색상
    - label_prefix: 레이블 접두사
    """
    for node_id, node in pose_graph.nodes.items():
        origin = node.pose[:3, 3]
        R = node.pose[:3, :3]
        colors = ["r", "g", "b"]
        for i in range(3):
            vec = R[:, i]
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                vec[0],
                vec[1],
                vec[2],
                color=colors[i],
                length=0.3,
                normalize=True,
            )
        ax.scatter(
            origin[0],
            origin[1],
            origin[2],
            s=50,
            label=(
                f"{label_prefix} {node_id}"
                if node_id == list(pose_graph.nodes.keys())[0]
                else ""
            ),
        )


def main():
    # PoseGraph 객체 생성
    graph = PoseGraph()

    # 1. 노드 A와 노드 B 추가
    graph.generate_node_at_graph("A", initial_pose=np.eye(4))  # 노드 A: 항등 행렬
    graph.generate_node_at_graph(
        "B", initial_pose=np.eye(4)
    )  # 노드 B: 초기 포즈는 항등 행렬

    # 2. 노드 B에 Prior Factor 추가: A에서 x로 1만큼 이동, 회전 없음
    prior_B = create_pose_matrix(0, 0, 0, np.array([1, 0, 0]))
    graph.add_prior_factor("B", prior_B)

    # 3. Between Residual 추가: A와 B 사이의 상대 포즈는 x로 0.5만큼 이동, 회전 없음
    between_AB = create_pose_matrix(0, 0, 0, np.array([0.5, 0, 0]))
    graph.add_between_factor("A", "B", between_AB)

    # 4. 초기 추정치에 노이즈 추가 (노드 B만)
    np.random.seed(42)  # 재현성을 위해 시드 고정
    initial_pose_B = graph.get_solution("B").copy()

    # 회전 노이즈 추가 (0.1 rad)
    noise_rot = 0.1 * np.random.randn(3)
    initial_pose_B[:3, :3] = oplus(initial_pose_B[:3, :3], noise_rot)

    # 평행 이동 노이즈 추가 (0.05 단위)
    noise_trans = 0.05 * np.random.randn(3)
    initial_pose_B[:3, 3] += noise_trans

    # 노드 B의 초기 추정치 업데이트
    graph.nodes["B"].pose = initial_pose_B

    # 5. 최적화 이전의 포즈 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_graph(graph, ax, color="b", label_prefix="Initial Pose")
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Before Optimization")
    ax.legend()
    plt.show()

    # 6. 그래프 최적화 수행
    graph.solve_graph()

    # 7. 최적화 이후의 포즈 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_graph(graph, ax, color="g", label_prefix="Optimized Pose")
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("After Optimization")
    ax.legend()
    plt.show()

    # 8. 최적화 결과 출력
    print("\nOptimized Poses:")
    for node_id in graph.nodes:
        pose = graph.get_solution(node_id)
        print(f"Node {node_id}:\n{pose}\n")


if __name__ == "__main__":
    main()
