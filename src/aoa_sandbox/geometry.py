import numpy as np


def get_mic_positions(array_type):
    if array_type == "square":
        return np.array([
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1, 0.1],
            [-0.1, 0.1]
        ])
    elif array_type == "linear":
        return np.array([[0.05*i, 0.0] for i in range(4)])
    else:
        raise ValueError(f"Unknown array type {array_type}")
