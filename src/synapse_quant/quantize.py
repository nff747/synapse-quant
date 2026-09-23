import numpy as np

def calculate_scale(tensor: np.ndarray) -> float:
    max_val = np.max(np.abs(tensor))
    if max_val == 0:
        return 1.0
    return 127.0 / max_val

def quantize_symmetric(tensor: np.ndarray, scale: float) -> np.ndarray:
    quantized = np.round(tensor * scale)
    quantized = np.clip(quantized, -127, 127)
    return quantized.astype(np.int8)
