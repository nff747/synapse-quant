import numpy as np

def test_int4_parity():
    weights = np.array([-7.0, -3.5, 0.0, 1.0, 3.5, 7.0, -1.0, 2.0], dtype=np.float32)
    scale = np.max(np.abs(weights)) / 7.0
    q = np.round(weights / scale).astype(np.int32)
    assert q[0] == -7
    assert q[5] == 7
    reconstructed = q * scale
    assert np.allclose(weights, reconstructed, atol=0.1)
