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
    롤, 피치, 요 (라디안 단위) 및 위치 좌표로부터 Pose3 객체를 생성합니다.
    """
    rotation = Rot3.RzRyRx(roll, pitch, yaw)
    translation = Point3(x, y, z)
    return Pose3(rotation, translation)


def add_noise_to_pose(pose, rot_noise_std=0.01, trans_noise_std=0.2):
    """
    Pose3 객체에 Gaussian 노이즈를 추가합니다.
    """
    # 회전에 노이즈 추가
    noise_rot = rot_noise_std * np.random.randn(3)  # 회전 벡터 노이즈
    # Rot3.Rx, Rot3.Ry, Rot3.Rz 사용
    noisy_rot = pose.rotation().compose(
        Rot3.Rx(noise_rot[0])
        .compose(Rot3.Ry(noise_rot[1]))
        .compose(Rot3.Rz(noise_rot[2]))
    )

    # 위치에 노이즈 추가
    noise_trans = trans_noise_std * np.random.randn(3)
    noisy_trans = Point3(
        pose.x() + noise_trans[0], pose.y() + noise_trans[1], pose.z() + noise_trans[2]
    )

    return Pose3(noisy_rot, noisy_trans)


def plot_poses(pose_list, ax, color="b", label_prefix="Pose"):
    """
    Pose3 객체 리스트를 3D 플롯에 시각화합니다.
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
        # 첫 번째 노드에만 레이블 추가하여 중복 방지
        if idx == 0:
            ax.scatter(
                origin[0],
                origin[1],
                origin[2],
                s=50,
                label=f"{label_prefix} {chr(ord('A') + idx)}",
            )
        else:
            ax.scatter(origin[0], origin[1], origin[2], s=50)
    # 레전드 중복 제거
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())


def main():
    # 팩터 그래프와 초기 추정치 초기화
    graph = NonlinearFactorGraph()
    initial = Values()

    # 노이즈 모델 정의
    # 매우 작은 노이즈 값은 수치적으로 불안정할 수 있으므로 약간 큰 값으로 설정
    prior_noise = noiseModel.Diagonal.Sigmas(
        np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    )
    odometry_noise = noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    )

    # 상대 이동 정의: x 방향으로 1 단위 전진 및 z축을 기준으로 90도 회전
    delta_x = 1.0
    delta_yaw = np.deg2rad(60)  # 90도를 라디안으로 변환

    # 노드 키 정의 (정수 키 사용)
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4

    # 포즈 A부터 E까지 생성
    poses = []
    for i in range(5):
        roll = 0
        pitch = 0
        yaw = delta_yaw * i
        x = delta_x * i
        y = 0
        z = 0
        pose = create_pose(roll, pitch, yaw, x, y, z)
        poses.append(pose)

    # 노드 A에 Prior Factor 추가
    graph.add(PriorFactorPose3(A, poses[0], prior_noise))

    # Between Factor 추가 (A-B, B-C, C-D, D-E)
    move_once = create_pose(0, 0, delta_yaw, delta_x, 0, 0)
    for i in range(4):
        from_key = i
        to_key = i + 1
        relative_pose = move_once
        print(f"relative_pose \n{relative_pose}")
        graph.add(BetweenFactorPose3(from_key, to_key, relative_pose, odometry_noise))

    # 초기 추정치에 노이즈 추가 및 삽입
    initial_estimates = []
    for i, key in enumerate([A, B, C, D, E]):
        noisy_pose = add_noise_to_pose(poses[i])
        initial.insert(key, noisy_pose)
        initial_estimates.append(noisy_pose)

    # 최적화 전 포즈 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(initial_estimates, ax, color="b", label_prefix="Initial Pose")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Before Optimization")
    plt.show()

    # 그래프 최적화 수행
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # 최적화된 포즈 추출
    optimized_poses = []
    for key in [A, B, C, D, E]:
        pose = result.atPose3(key)
        optimized_poses.append(pose)

    # 최적화 후 포즈 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(optimized_poses, ax, color="g", label_prefix="Optimized Pose")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("After Optimization")
    plt.show()

    # 최적화된 포즈 출력
    print("\nOptimized Poses:")
    for i, key in enumerate([A, B, C, D, E]):
        pose = result.atPose3(key)
        print(f"Node {chr(ord('A') + i)}:\n{pose}\n")


if __name__ == "__main__":
    main()
