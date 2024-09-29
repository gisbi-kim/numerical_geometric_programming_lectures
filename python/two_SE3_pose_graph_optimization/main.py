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
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
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

def plot_poses(poses_initial, poses_optimized, ax):
    """
    초기 포즈와 최적화된 포즈를 3D 플롯에 시각화합니다.
    """
    for idx, (T_init, T_opt) in enumerate(zip(poses_initial, poses_optimized)):
        # 초기 포즈
        origin_init = T_init[:3, 3]
        R_init = T_init[:3, :3]
        ax.quiver(origin_init[0], origin_init[1], origin_init[2],
                  R_init[0,0], R_init[1,0], R_init[2,0],
                  color='r', length=0.3, normalize=True)
        ax.quiver(origin_init[0], origin_init[1], origin_init[2],
                  R_init[0,1], R_init[1,1], R_init[2,1],
                  color='g', length=0.3, normalize=True)
        ax.quiver(origin_init[0], origin_init[1], origin_init[2],
                  R_init[0,2], R_init[1,2], R_init[2,2],
                  color='b', length=0.3, normalize=True)
        ax.scatter(origin_init[0], origin_init[1], origin_init[2], color='k', s=50, label=f"Initial Pose {idx+1}" if idx == 0 else "")
        
        # 최적화된 포즈
        origin_opt = T_opt[:3, 3]
        R_opt = T_opt[:3, :3]
        ax.quiver(origin_opt[0], origin_opt[1], origin_opt[2],
                  R_opt[0,0], R_opt[1,0], R_opt[2,0],
                  color='r', length=0.3, normalize=True, linestyle='dashed')
        ax.quiver(origin_opt[0], origin_opt[1], origin_opt[2],
                  R_opt[0,1], R_opt[1,1], R_opt[2,1],
                  color='g', length=0.3, normalize=True, linestyle='dashed')
        ax.quiver(origin_opt[0], origin_opt[1], origin_opt[2],
                  R_opt[0,2], R_opt[1,2], R_opt[2,2],
                  color='b', length=0.3, normalize=True, linestyle='dashed')
        ax.scatter(origin_opt[0], origin_opt[1], origin_opt[2], color='c', s=50, label=f"Optimized Pose {idx+1}" if idx == 0 else "")
    
    ax.legend()

def main():
    # 1. 노드 정의 (2개 노드: A와 B)
    # 노드 A: 항등 행렬 (Prior)
    T_A_prior = np.eye(4)
    
    # 노드 B: 노드 A에서 x축으로 1만큼 이동, 회전 없음 (Prior)
    T_B_prior = create_pose_matrix(0, 0, 0, np.array([1, 0, 0]))
    
    # 초기 추정치 (약간의 노이즈 추가)
    np.random.seed(42)  # 재현성을 위해 시드 고정
    T_A_est = T_A_prior.copy()
    T_B_est = T_B_prior.copy()
    
    # 노드 B에 회전 노이즈 추가 (0.1 rad)
    noise_rot = 0.1 * np.random.randn(3)
    T_B_est[:3, :3] = oplus(T_B_est[:3, :3], noise_rot)
    
    # 노드 B에 평행 이동 노이즈 추가 (0.05 단위)
    noise_trans = 0.05 * np.random.randn(3)
    T_B_est[:3, 3] += noise_trans
    
    poses_initial = [T_A_est.copy(), T_B_est.copy()]
    poses_optimized = [T_A_est.copy(), T_B_est.copy()]
    
    # 2. Prior Factors 설정
    priors = [T_A_prior.copy(), T_B_prior.copy()]
    
    # 3. Between Residual 설정
    # Between Residual: 노드 A와 노드 B 사이의 상대 포즈는 x축으로 0.5만큼 이동, 회전 없음
    T_AB_target = create_pose_matrix(0, 0, 0, np.array([0.9, 0, 0]))
    between_edges = [
        (0, 1, T_AB_target.copy())  # (노드 A, 노드 B, 상대 포즈)
    ]
    
    # 4. 최적화 파라미터 설정
    max_iterations = 100
    tolerance = 1e-6
    
    # 5. 최적화 루프 (Gauss-Newton)
    for iteration in range(max_iterations):
        H = np.zeros((6 * 2, 6 * 2))  # 2 노드, 각 노드마다 6 DOF
        b = np.zeros(6 * 2)
        
        # Prior Residuals
        for i, T_prior in enumerate(priors):
            T_est = poses_optimized[i]
            # 회전 Residual
            R_residual = np.dot(T_prior[:3, :3].T, T_est[:3, :3])
            dtheta = Log_map(R_residual)
            # 평행 이동 Residual
            dt = T_est[:3, 3] - T_prior[:3, 3]
            residual = np.hstack((dtheta, dt))
            
            # Residual Weighting (Prior의 경우 가중치 1)
            weight_prior = 1.0
            residual_weighted = residual * np.sqrt(weight_prior)
            
            # Jacobian은 단위 행렬 (독립적)
            J = np.eye(6)
            J_weighted = J * np.sqrt(weight_prior)
            
            # Hessian 및 b 업데이트
            H_i = J_weighted.T @ J_weighted
            b_i = J_weighted.T @ residual_weighted
            
            H[6*i:6*(i+1), 6*i:6*(i+1)] += H_i
            b[6*i:6*(i+1)] += b_i
        
        # Between Residuals
        for edge in between_edges:
            i, j, T_AB = edge
            T_i = poses_optimized[i]
            T_j = poses_optimized[j]
            
            # 상대 Residual 계산: inv(T_i) * T_j * inv(T_AB)
            T_residual = np.linalg.inv(T_i) @ T_j @ np.linalg.inv(T_AB)
            R_residual = T_residual[:3, :3]
            dtheta = Log_map(R_residual)
            dt = T_residual[:3, 3]
            residual = np.hstack((dtheta, dt))
            
            # Residual Weighting (Between Residual의 경우 가중치 0.5)
            weight_between = 0.5
            residual_weighted = residual * np.sqrt(weight_between)
            
            # Jacobian 설정
            # 노드 i는 -I, 노드 j는 +I (간단화된 가정)
            J_i = -np.eye(6)
            J_j = np.eye(6)
            J_i_weighted = J_i * np.sqrt(weight_between)
            J_j_weighted = J_j * np.sqrt(weight_between)
            
            # Hessian 및 b 업데이트
            H_i = J_i_weighted.T @ J_i_weighted
            H_j = J_j_weighted.T @ J_j_weighted
            H_ij = J_i_weighted.T @ J_j_weighted
            
            H[6*i:6*(i+1), 6*i:6*(i+1)] += H_i
            H[6*j:6*(j+1), 6*j:6*(j+1)] += H_j
            H[6*i:6*(i+1), 6*j:6*(j+1)] += H_ij
            H[6*j:6*(j+1), 6*i:6*(i+1)] += H_ij.T
            
            b[6*i:6*(i+1)] += J_i_weighted.T @ residual_weighted
            b[6*j:6*(j+1)] += J_j_weighted.T @ residual_weighted
        
        # Gauss-Newton 업데이트: H * dx = -b
        try:
            dx = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered during optimization.")
            break
        
        # 수렴 조건 확인
        if np.linalg.norm(dx) < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        
        # 포즈 업데이트
        for i in range(2):
            tangent_vec = dx[6*i:6*(i+1)]
            poses_optimized[i] = oplus(poses_optimized[i], tangent_vec)
    
    # 최적화 완료 메시지
    print("Optimization completed.")
    
    # 6. 결과 출력
    print("\nTrue Poses:")
    print("Pose A (True):\n", T_A_prior)
    print("Pose B (True):\n", T_B_prior)
    
    print("\nInitial Estimated Poses:")
    print("Pose A (Initial):\n", poses_initial[0])
    print("Pose B (Initial):\n", poses_initial[1])
    
    print("\nOptimized Estimated Poses:")
    print("Pose A (Optimized):\n", poses_optimized[0])
    print("Pose B (Optimized):\n", poses_optimized[1])
    
    # 7. 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    plot_poses(poses_initial, poses_optimized, ax)
    
    # 그래프 설정
    ax.set_xlim([-0.5, 2])
    ax.set_ylim([-0.5, 2])
    ax.set_zlim([-0.5, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Pose Graph Optimization: Initial vs Optimized Poses")
    plt.show()

if __name__ == "__main__":
    main()
