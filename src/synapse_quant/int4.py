import numpy as np

def dequantize_int4(packed: np.ndarray, length: int, scale: float) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    for i in range(length):
        u32_idx = i // 8
        shift = (i % 8) * 4
        raw = (int(packed[u32_idx]) >> shift) & 0x0F
        signed_val = raw - 16 if raw >= 8 else raw
        out[i] = float(signed_val) * scale
    return out
