import os
import sys
import numpy as np  # Ensure numpy is imported

# from scipy.linalg import logm, expm  # For Log_map and oplus functions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.spatial.transform import Rotation as R
from lib.pose_module import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_rotation_matrix(roll, pitch, yaw):
    """
    롤, 피치, 요 각을 사용하여 회전 행렬을 생성합니다.
    """
    # R.from_euler()를 사용하여 회전 행렬 생성
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    return rotation.as_matrix()  # 회전 행렬로 변환하여 반환


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

        self.set_weights()

    def set_weights(self):
        self.rot_trans_scale_ratio = 1

        self.weight_rot_prior = 100000000.0    # Prior에서 회전에 대한 가중치
        self.weight_trans_prior = 100000000.0  # Prior에서 위치에 대한 가중치
 
        self.weight_rot_between = 1.0   # Between에서 회전에 대한 가중치
        self.weight_trans_between = 1.0 # Between에서 위치에 대한 가중치

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

    def solve_graph(self, max_iterations=300, tolerance=1e-3):
        """
        포즈 그래프를 최적화합니다.

        Parameters:
        - max_iterations: 최적화의 최대 반복 횟수
        - tolerance: 수렴을 판단할 허용 오차
        """
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        num_nodes = len(node_ids)

        iteration_msg_list = [] 
        for iteration in range(max_iterations):
            total_residual = 0.0  # total residual 초기화

            # --- 1. 회전 업데이트 단계 ---
            # 회전에 관련된 Hessian과 b 벡터 초기화
            H_rot = np.zeros((3 * num_nodes, 3 * num_nodes))
            b_rot = np.zeros(3 * num_nodes)

            # Prior Residuals for Rotation
            for prior in self.prior_factors:
                i = node_id_to_idx[prior.node_id]
                T_est = self.nodes[prior.node_id].pose
                T_prior = prior.prior_pose

                # Residual 계산: log_map(inv(T_prior) * T_est)
                T_residual = np.linalg.inv(T_prior) @ T_est
                R_residual = T_residual[:3, :3]
                drotvec = Log_map(R_residual)

                # Accumulate total residual
                total_residual += np.linalg.norm(drotvec)

                # Jacobian은 단위 행렬 (Prior Residual은 노드 자체에만 영향)
                J_rot = np.eye(3)

                # 회전에 대한 가중치 적용
                W_rot_prior = np.sqrt(self.weight_rot_prior) * np.eye(3)

                residual_rot_weighted = W_rot_prior @ drotvec
                J_rot_weighted = W_rot_prior @ J_rot

                # Hessian 및 b 업데이트
                H_rot_i = J_rot_weighted.T @ J_rot_weighted
                b_rot_i = J_rot_weighted.T @ residual_rot_weighted

                H_rot[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] += H_rot_i
                b_rot[3 * i : 3 * (i + 1)] += b_rot_i

            # Between Residuals for Rotation
            for between in self.between_factors:
                i = node_id_to_idx[between.from_node_id]
                j = node_id_to_idx[between.to_node_id]
                T_i = self.nodes[between.from_node_id].pose
                T_j = self.nodes[between.to_node_id].pose
                T_AB = between.relative_pose

                # Residual 계산: log_map(inv(T_i) * T_j * inv(T_AB))
                T_est = np.linalg.inv(T_i) @ T_j
                T_residual = T_est @ np.linalg.inv(T_AB)
                R_residual = T_residual[:3, :3]
                drotvec = Log_map(R_residual)

                # Optional: 안전한 업데이트 적용
                use_safe_update = True 
                if use_safe_update:
                    while 1e-2 < np.max(np.abs(drotvec)):
                        drotvec = drotvec * 0.5  # 더 작은 스텝으로 조정

                residual_rot = drotvec
                total_residual += np.linalg.norm(residual_rot)

                # Jacobian 설정: 노드 i는 -I, 노드 j는 +I
                J_i_rot = -np.eye(3)
                J_j_rot = np.eye(3)

                # 회전에 대한 가중치 적용
                W_rot_between = np.sqrt(self.weight_rot_between) * np.eye(3)

                residual_rot_weighted = W_rot_between @ residual_rot
                J_i_rot_weighted = W_rot_between @ J_i_rot
                J_j_rot_weighted = W_rot_between @ J_j_rot

                # Hessian 및 b 업데이트
                H_rot_ii = J_i_rot_weighted.T @ J_i_rot_weighted
                H_rot_jj = J_j_rot_weighted.T @ J_j_rot_weighted
                H_rot_ij = J_i_rot_weighted.T @ J_j_rot_weighted

                H_rot[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] += H_rot_ii
                H_rot[3 * j : 3 * (j + 1), 3 * j : 3 * (j + 1)] += H_rot_jj
                H_rot[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] += H_rot_ij
                H_rot[3 * j : 3 * (j + 1), 3 * i : 3 * (i + 1)] += H_rot_ij.T

                b_rot[3 * i : 3 * (i + 1)] += J_i_rot_weighted.T @ residual_rot_weighted
                b_rot[3 * j : 3 * (j + 1)] += J_j_rot_weighted.T @ residual_rot_weighted

            # 회전 업데이트 해를 계산 (H_rot * drot = -b_rot)
            try:
                damping_rot = 0.000
                H_rot_damped = H_rot + damping_rot * np.eye(H_rot.shape[0])
                drot = np.linalg.solve(H_rot_damped, -b_rot)
            except np.linalg.LinAlgError:
                print("Singular rotation matrix encountered during optimization.")
                break

            # 회전 업데이트 적용
            for node_id in node_ids:
                idx = node_id_to_idx[node_id]
                delta_rot = drot[3 * idx : 3 * (idx + 1)]
                if np.linalg.norm(delta_rot) > 1e-6:
                    rotation_update = R.from_rotvec(delta_rot).as_matrix()
                    self.nodes[node_id].pose[:3, :3] = rotation_update @ self.nodes[node_id].pose[:3, :3]

            # --- 2. 위치 업데이트 단계 ---
            # 위치에 관련된 Hessian과 b 벡터 초기화
            H_trans = np.zeros((3 * num_nodes, 3 * num_nodes))
            b_trans = np.zeros(3 * num_nodes)

            # Prior Residuals for Translation
            for prior in self.prior_factors:
                i = node_id_to_idx[prior.node_id]
                T_est = self.nodes[prior.node_id].pose
                T_prior = prior.prior_pose

                # Residual 계산: inv(T_prior) * T_est의 translation 부분
                T_residual = np.linalg.inv(T_prior) @ T_est
                dpos = self.rot_trans_scale_ratio * T_residual[:3, 3]

                # Accumulate total residual
                total_residual += np.linalg.norm(dpos)

                # Jacobian은 단위 행렬 (Prior Residual은 노드 자체에만 영향)
                J_trans = np.eye(3)

                # 위치에 대한 가중치 적용
                W_trans_prior = np.sqrt(self.weight_trans_prior) * np.eye(3)

                residual_trans_weighted = W_trans_prior @ dpos
                J_trans_weighted = W_trans_prior @ J_trans

                # Hessian 및 b 업데이트
                H_trans_i = J_trans_weighted.T @ J_trans_weighted
                b_trans_i = J_trans_weighted.T @ residual_trans_weighted

                H_trans[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] += H_trans_i
                b_trans[3 * i : 3 * (i + 1)] += b_trans_i

            # Between Residuals for Translation
            for between in self.between_factors:
                i = node_id_to_idx[between.from_node_id]
                j = node_id_to_idx[between.to_node_id]
                T_i = self.nodes[between.from_node_id].pose
                T_j = self.nodes[between.to_node_id].pose
                T_AB = between.relative_pose

                # Residual 계산: inv(T_i) * T_j * inv(T_AB)의 translation 부분
                T_est = np.linalg.inv(T_i) @ T_j
                T_residual = T_est @ np.linalg.inv(T_AB)
                dpos = self.rot_trans_scale_ratio * T_residual[:3, 3]

                # Optional: 안전한 업데이트 적용
                use_safe_update = True 
                if use_safe_update:
                    while 1e-1 < np.max(np.abs(dpos)):
                        dpos = dpos * 0.5  # 더 작은 스텝으로 조정

                residual_trans = dpos
                total_residual += np.linalg.norm(residual_trans)

                # Jacobian 설정: 노드 i는 -I, 노드 j는 +I
                J_i_trans = -np.eye(3)
                J_j_trans = np.eye(3)

                # 위치에 대한 가중치 적용
                W_trans_between = np.sqrt(self.weight_trans_between) * np.eye(3)

                residual_trans_weighted = W_trans_between @ residual_trans
                J_i_trans_weighted = W_trans_between @ J_i_trans
                J_j_trans_weighted = W_trans_between @ J_j_trans

                # Hessian 및 b 업데이트
                H_trans_ii = J_i_trans_weighted.T @ J_i_trans_weighted
                H_trans_jj = J_j_trans_weighted.T @ J_j_trans_weighted
                H_trans_ij = J_i_trans_weighted.T @ J_j_trans_weighted

                H_trans[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] += H_trans_ii
                H_trans[3 * j : 3 * (j + 1), 3 * j : 3 * (j + 1)] += H_trans_jj
                H_trans[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] += H_trans_ij
                H_trans[3 * j : 3 * (j + 1), 3 * i : 3 * (i + 1)] += H_trans_ij.T

                b_trans[3 * i : 3 * (i + 1)] += J_i_trans_weighted.T @ residual_trans_weighted
                b_trans[3 * j : 3 * (j + 1)] += J_j_trans_weighted.T @ residual_trans_weighted

            # 위치 업데이트 해를 계산 (H_trans * dtrans = -b_trans)
            try:
                damping_trans = 0.000
                H_trans_damped = H_trans + damping_trans * np.eye(H_trans.shape[0])
                dtrans = np.linalg.solve(H_trans_damped, -b_trans)
            except np.linalg.LinAlgError:
                print("Singular translation matrix encountered during optimization.")
                break

            # 위치 업데이트 적용
            for node_id in node_ids:
                idx = node_id_to_idx[node_id]
                delta_trans = dtrans[3 * idx : 3 * (idx + 1)]
                self.nodes[node_id].pose[:3, 3] += delta_trans

            # 업데이트의 노름 계산하여 수렴 여부 확인
            norm_drot = np.linalg.norm(drot)
            norm_dtrans = np.linalg.norm(dtrans)
            norm_dx = norm_drot + norm_dtrans
            iteration_msg = f"Iteration {iteration}: |drot| = {norm_drot:.6f}, |dtrans| = {norm_dtrans:.6f}, total_residual = {total_residual:.6f}"
            iteration_msg_list.append(iteration_msg)
            print(iteration_msg)

            if norm_dx < tolerance:
                print(f"Converged at iteration {iteration}")
                break

            if iteration == max_iterations - 1:
                print("Reached maximum iterations without convergence.")

        print("Optimization done.\n")
        for msg in iteration_msg_list:
            print(msg)

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
            ),
        )


def main():
    # PoseGraph 객체 생성
    graph = PoseGraph()

    delta_yaw = np.deg2rad(90)  # 90도 회전을 라디안으로 변환

    # 1. 노드 A, B, C, D, E 추가 (네모를 형성하는 5개 노드)
    initial_pose_A = create_pose_matrix(0, 0, 0, np.array([0, 0, 0]))
    graph.generate_node_at_graph("A", initial_pose=initial_pose_A)

    move_once = create_pose_matrix(0, 0, delta_yaw, np.array([1, 0, 0]))
    initial_pose_B = initial_pose_A @ move_once
    graph.generate_node_at_graph("B", initial_pose=initial_pose_B)

    initial_pose_C = initial_pose_B @ move_once
    graph.generate_node_at_graph("C", initial_pose=initial_pose_C)

    initial_pose_D = initial_pose_C @ move_once
    graph.generate_node_at_graph("D", initial_pose=initial_pose_D)

    initial_pose_E = initial_pose_D @ move_once
    graph.generate_node_at_graph("E", initial_pose=initial_pose_E)

    # 2. Prior Factors 추가
    prior_A = create_pose_matrix(0, 0, 0, np.array([0, 0, 0]))
    graph.add_prior_factor("A", prior_A)

    if 0:
        graph.add_prior_factor("B", initial_pose_B)
        graph.add_prior_factor("C", initial_pose_C)
        graph.add_prior_factor("D", initial_pose_D)
        graph.add_prior_factor("E", initial_pose_E)

    # 3. 초기 추정치에 노이즈 추가
    for node_id in ["B", "C", "D", "E"]:
        initial_pose = graph.get_solution(node_id).copy()

        # 회전 노이즈 추가 (0.1 rad)
        # noise_rot = 0.01 * np.random.randn(3)
        # noise_trans = 0.01 * np.random.randn(3)
        noise_rot = 0.01 * np.random.randn(3)
        noise_trans = 0.2 * np.random.randn(3)
        
        initial_rotation = R.from_matrix(initial_pose[:3, :3])
        noisy_rotation = initial_rotation * R.from_rotvec(noise_rot)
        initial_pose[:3, :3] = noisy_rotation.as_matrix()
        initial_pose[:3, 3] += noise_trans

        # 노드의 초기 추정치 업데이트
        graph.nodes[node_id].pose = initial_pose

    # 4. Between Factors 추가 (각 Between Factor에 90도 회전 추가)
    if 1:
        between_AB = move_once
        graph.add_between_factor("A", "B", between_AB)

        between_BC = move_once
        graph.add_between_factor("B", "C", between_BC)

        between_CD = move_once
        graph.add_between_factor("C", "D", between_CD)

        between_DE = move_once
        graph.add_between_factor("D", "E", between_DE)

        between_CE = move_once @ move_once
        graph.add_between_factor("C", "E", between_CE)

        # between_AE = initial_pose_E @ np.linalg.inv(initial_pose_A)
        # graph.add_between_factor("A", "E", between_AE)

    # 5. 최적화 이전의 포즈 시각화
    ax_lim = 5.0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_graph(graph, ax, color="b", label_prefix="Initial Pose")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_zlim([-ax_lim, ax_lim])
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
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_zlim([-ax_lim, ax_lim])
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
