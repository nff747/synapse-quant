import pytest
import numpy as np
from synapse_quant.quantize import calculate_scale

def test_calculate_scale():
    tensor = np.array([-2.0, 1.0, 4.0], dtype=np.float32)
    scale = calculate_scale(tensor)
    assert np.isclose(scale, 127.0 / 4.0)
