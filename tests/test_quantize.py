import pytest
import numpy as np
from synapse_quant.quantize import calculate_scale

def test_calculate_scale():
    tensor = np.array([-2.0, 1.0, 4.0], dtype=np.float32)
    scale = calculate_scale(tensor)
    assert np.isclose(scale, 127.0 / 4.0)

def test_quantize_symmetric():
    from synapse_quant.quantize import quantize_symmetric
    tensor = np.array([-2.0, 1.0, 4.0], dtype=np.float32)
    scale = calculate_scale(tensor)
    q_tensor = quantize_symmetric(tensor, scale)
    assert q_tensor.dtype == np.int8
    assert q_tensor[2] == 127

def test_dequantize_symmetric():
    from synapse_quant.quantize import quantize_symmetric, calculate_scale
    from synapse_quant.dequantize import dequantize_symmetric
    tensor = np.array([-2.0, 1.0, 4.0], dtype=np.float32)
    scale = calculate_scale(tensor)
    q_tensor = quantize_symmetric(tensor, scale)
    dq_tensor = dequantize_symmetric(q_tensor, scale)
    
    assert np.allclose(tensor, dq_tensor, atol=0.05)
