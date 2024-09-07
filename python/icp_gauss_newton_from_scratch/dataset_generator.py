import numpy as np


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
