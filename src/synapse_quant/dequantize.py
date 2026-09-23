import numpy as np

def dequantize_symmetric(quantized_tensor: np.ndarray, scale: float) -> np.ndarray:
    return (quantized_tensor.astype(np.float32) / scale)
