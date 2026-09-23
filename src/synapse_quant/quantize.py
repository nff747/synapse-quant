import numpy as np

def calculate_scale(tensor: np.ndarray) -> float:
    max_val = np.max(np.abs(tensor))
    if max_val == 0:
        return 1.0
    return 127.0 / max_val
