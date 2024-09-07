import numpy as np

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
