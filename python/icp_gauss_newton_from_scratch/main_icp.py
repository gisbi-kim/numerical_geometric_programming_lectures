import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataset_generator import generate_lidar_data
from visualizer import visualize_iterations, visualize_results
from config import config


def objective_function(params, source, target):
    theta, tx, ty = params
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T = np.array([tx, ty])
    transformed = np.dot(source, R.T) + T
    return transformed - target


def compute_jacobian(params, source):
    theta, _, _ = params
    n_points = source.shape[0]
    J = np.zeros((n_points, 2, 3))

    # Jacobian for rotation
    J[:, 0, 0] = -source[:, 0] * np.sin(theta) - source[:, 1] * np.cos(theta)
    J[:, 1, 0] = source[:, 0] * np.cos(theta) - source[:, 1] * np.sin(theta)

    # Jacobian for translation
    J[:, 0, 1] = 1  # dx
    J[:, 1, 2] = 1  # dy

    return J


def gauss_newton_step(params, source, target):
    residuals = objective_function(params, source, target)
    J = compute_jacobian(params, source)

    JTJ = np.zeros((3, 3))
    JTr = np.zeros(3)

    for i in range(3):
        for j in range(3):
            JTJ[i, j] = np.sum(J[:, :, i] * J[:, :, j])
        JTr[i] = np.sum(J[:, :, i] * residuals)

    optimized_error_state = np.linalg.solve(JTJ, -JTr)

    cost_for_debug = np.sum(residuals**2)
    return optimized_error_state, cost_for_debug


def icp(
    source,
    target,
    apply_pretranslation=True,
    apply_prerotation=True,
    using_pruning=True,
    pruning_threshold=0.999,
    max_iterations=50,
    tolerance=0.005,
):
    current_source = source.copy()
    target_with_initial_guess = target.copy()

    transformations = []
    for iteration in range(max_iterations):
        print(f"Iteration {iteration}")

        if apply_pretranslation:
            target_center_moved_wrt_source = np.mean(target, axis=0) - np.mean(
                current_source, axis=0
            )
            target_with_initial_guess = target - target_center_moved_wrt_source

        if apply_prerotation:
            # Estimate initial rotation using principal component analysis (PCA)
            source_centered = current_source - np.mean(current_source, axis=0)
            target_centered_pca = target_with_initial_guess - np.mean(
                target_with_initial_guess, axis=0
            )

            # Compute covariance matrices
            cov_source = np.cov(source_centered.T)
            cov_target = np.cov(target_centered_pca.T)

            # Compute eigenvectors
            _, v_source = np.linalg.eig(cov_source)
            _, v_target = np.linalg.eig(cov_target)

            # Estimate rotation matrix
            R_init = np.dot(v_target, v_source.T)

            # Apply initial rotation to current_source
            current_source = np.dot(
                current_source - np.mean(current_source, axis=0), R_init.T
            ) + np.mean(target_with_initial_guess, axis=0)

            # Recenter target_centered
            target_with_initial_guess = (
                target_with_initial_guess
                - np.mean(target_with_initial_guess, axis=0)
                + np.mean(current_source, axis=0)
            )

        # find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
            target_with_initial_guess
        )
        distances, indices = nbrs.kneighbors(current_source)

        # Gauss-Newton optimization
        params = np.array(
            [0.0, 0.0, 0.0], dtype=np.float64
        )  # Initial guess for [theta, tx, ty]
        for _ in range(10):  # Inner optimization loop
            optimized_error_state, cost_for_debug_sum = gauss_newton_step(
                params, current_source, target[indices.ravel()]
            )
            params += optimized_error_state
            if np.linalg.norm(optimized_error_state) < 1e-6:
                break

        theta, tx, ty = params
        print(f"theta: {theta:.4f}, tx: {tx:.4f}, ty: {ty:.4f}")
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        T = np.array([tx, ty])
        current_source = np.dot(current_source, R.T) + T

        transformations.append(current_source.copy())

        # Check convergence
        cost_for_debug_avg = cost_for_debug_sum / len(current_source)
        delta_transformation = np.linalg.norm(np.array([tx, ty]))
        print(
            f"Current delta_transformation: {delta_transformation:.4f} (tolerance: {tolerance})"
        )
        if delta_transformation < tolerance:
            print(f"Converged at iteration {iteration}")
            print(f"Current residual: {cost_for_debug_avg:.4f}")
            break

        # if some points have big residual, remove it for the next iteration
        # do not use the fixed value, but the 98% biggest of the point wise residual
        point_wise_residual = np.linalg.norm(
            current_source - target[indices.ravel()], axis=1
        )
        if using_pruning:
            big_residual_indices = point_wise_residual > np.quantile(
                point_wise_residual, pruning_threshold
            )
            current_source = current_source[~big_residual_indices]

    # if not converged, print message
    if iteration == max_iterations - 1:
        print(f"Current residual: {cost_for_debug_avg:.4f}")
        print("Max iteration reached.")

    # if residual is under some threshold, then it is converged
    if cost_for_debug_avg < 1e-1:
        print(f"Converged at iteration {iteration}")
        print(f"Current residual: {cost_for_debug_avg:.4f}")

    return current_source, transformations


def transform_points(points, angle_rad, translation):
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.dot(points, rotation_matrix.T) + translation


def experiment():

    source = generate_lidar_data(
        config["num_points"],
        config["noise_level"],
        config["big_noised_ratio"],
        config["shape"],
    )
    target = generate_lidar_data(
        config["num_points"],
        config["noise_level"],
        config["big_noised_ratio"],
        config["shape"],
    )
    target = transform_points(
        points=target,
        angle_rad=np.deg2rad(config["angle_deg"]),
        translation=config["translation"],
    )

    aligned, transformations = icp(
        source,
        target,
        using_pruning=config["using_pruning"],
        pruning_threshold=config["pruning_threshold"],
        apply_pretranslation=config["apply_pretranslation"],
        apply_prerotation=config["apply_prerotation"],
        max_iterations=config["max_iterations"],
    )

    visualize_iterations(target, transformations, source, aligned)
    visualize_results(source, target, aligned)


if __name__ == "__main__":
    experiment()
