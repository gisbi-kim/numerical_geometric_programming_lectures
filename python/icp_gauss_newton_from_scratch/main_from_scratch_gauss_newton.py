import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


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

    delta_solution = np.linalg.solve(JTJ, -JTr)
    cost_for_debug = np.sum(residuals**2)
    return delta_solution, cost_for_debug


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
            delta_solution, cost_for_debug_sum = gauss_newton_step(
                params, current_source, target[indices.ravel()]
            )
            params += delta_solution
            if np.linalg.norm(delta_solution) < 1e-6:
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
        point_wise_residual = np.linalg.norm(current_source - target[indices.ravel()], axis=1)
        if using_pruning:
            big_residual_indices = point_wise_residual >  np.quantile(point_wise_residual, pruning_threshold) 
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


def generate_lidar_data(
    num_points=1000, noise_level=0.1, big_noised_ratio=0.1, shape="square"
):

    if shape == "circle":
        theta = np.linspace(0, 2 * np.pi, num_points)
        r = 5 + np.random.randn(num_points) * noise_level
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack((x, y))
    elif shape == "square":
        room_width, room_length, corridor_width = 10, 15, 2
        walls = [
            np.column_stack(
                (
                    np.zeros(num_points // 4),
                    np.linspace(0, room_length, num_points // 4),
                )
            ),
            np.column_stack(
                (
                    np.full(num_points // 4, room_width),
                    np.linspace(0, room_length, num_points // 4),
                )
            ),
            np.column_stack(
                (
                    np.linspace(0, room_width, num_points // 4),
                    np.full(num_points // 4, room_length),
                )
            ),
            np.column_stack(
                (np.linspace(0, room_width, num_points // 4), np.zeros(num_points // 4))
            ),
        ]
        corridor = [
            np.column_stack(
                (
                    np.full(num_points // 8, (room_width - corridor_width) / 2),
                    np.linspace(0, room_length, num_points // 8),
                )
            ),
            np.column_stack(
                (
                    np.full(num_points // 8, (room_width + corridor_width) / 2),
                    np.linspace(0, room_length, num_points // 8),
                )
            ),
        ]
        points = np.vstack(walls + corridor)
    elif shape == "usa":
        # Simplified USA outline (lower 48 states)
        usa_outline = np.array(
            [
                [0, 0],
                [2, 1],
                [4, 1],
                [5, 2],
                [7, 2],
                [8, 1],
                [9, 2],
                [10, 1],
                [10, 3],
                [9, 4],
                [9, 5],
                [8, 6],
                [7, 6],
                [6, 7],
                [4, 7],
                [3, 6],
                [2, 6],
                [1, 5],
                [0, 4],
                [0, 0],
            ]
        )

        # Scale the outline
        usa_outline *= 2

        # Interpolate points along the outline
        num_segments = len(usa_outline) - 1
        points_per_segment = num_points // num_segments

        interpolated_points = []
        for i in range(num_segments):
            start = usa_outline[i]
            end = usa_outline[i + 1]
            segment_points = np.linspace(start, end, points_per_segment, endpoint=False)
            interpolated_points.extend(segment_points)

        points = np.array(interpolated_points)

        # Add noise
        points += np.random.normal(0, noise_level, points.shape)

    # big_noised_ratio percent of points are outliers
    num_big_noised = int(len(points) * big_noised_ratio)
    print(f"num_big_noised: {num_big_noised} / {len(points)}")

    # Select outlier points
    big_noised_indices = np.random.choice(len(points), num_big_noised, replace=False)
    big_noised_points = points[big_noised_indices]
    big_noised_points[:, 0] += np.random.normal(0, 10 * noise_level, num_big_noised)
    big_noised_points[:, 1] += np.random.normal(0, 10 * noise_level, num_big_noised)

    points = np.concatenate([points, big_noised_points])

    return points


def transform_points(points, angle_rad, translation):
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.dot(points, rotation_matrix.T) + translation


def visualize_iterations(target, transformations, source, aligned):
    num_iterations = min(100, len(transformations))

    # Add more vertical space between rows
    fig, axes = plt.subplots(4, 6, figsize=(20, 25), gridspec_kw={"hspace": 0.3})

    fig.suptitle("ICP Algorithm Iterations", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_iterations:
            ax.scatter(
                target[:, 0], target[:, 1], c="b", label="Target", alpha=0.5, s=1
            )
            ax.scatter(
                transformations[i][:, 0],
                transformations[i][:, 1],
                c="r",
                label="Source",
                alpha=0.5,
                s=1,
            )

            ax.set_title(f"Iteration {i+1}")
            ax.legend(fontsize="x-small")
        else:
            ax.axis("off")

    for ax in axes.flat:
        ax.set_aspect("equal", "box")
        ax.tick_params(axis="both", which="major", labelsize=6)

    plt.tight_layout()
    plt.show()


def visualize_results(source, target, aligned):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.scatter(source[:, 0], source[:, 1], c="r", label="Source")
    ax1.scatter(target[:, 0], target[:, 1], c="b", label="Target")
    ax1.legend()
    ax1.set_title("Before Alignment")
    ax1.axis("equal")

    ax2.scatter(aligned[:, 0], aligned[:, 1], c="r", label="Aligned Source")
    ax2.scatter(target[:, 0], target[:, 1], c="b", label="Target")
    ax2.legend()
    ax2.set_title("After Alignment")
    ax2.axis("equal")

    ax3.scatter(source[:, 0], source[:, 1], c="r", alpha=0.5, label="Original Source")
    ax3.scatter(target[:, 0], target[:, 1], c="b", alpha=0.5, label="Target")
    ax3.scatter(aligned[:, 0], aligned[:, 1], c="g", alpha=0.5, label="Aligned Source")
    ax3.legend()
    ax3.set_title("Comparison")
    ax3.axis("equal")

    plt.tight_layout()
    plt.show()


def slam_simulation():

    config = {
        "num_points": 3000,
        "noise_level": 0.05,
        "big_noised_ratio": 0.5,  # these points will be noised by 10 * noise_level
        "shape": "square",  # "circle" or "square" or "usa"
        "angle_deg": 35,
        "max_iterations": 200,
        "translation": np.array([2.5, 1.5]),
        "apply_pretranslation": True,
        "apply_prerotation": True,
        "using_pruning": True,       
        "pruning_threshold": 0.99,
    }

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
    slam_simulation()
