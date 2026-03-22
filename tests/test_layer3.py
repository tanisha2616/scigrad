import unittest
import numpy as np
import os

# Skip tests if OpenCL is not available or forced to CPU
try:
    import pyopencl as cl
    from scigrad.device import Device
    from scigrad.codegen.opencl import OpenCLBackend
    OPENCL_AVAILABLE = isinstance(Device, OpenCLBackend)
except Exception:
    OPENCL_AVAILABLE = False

from scigrad.tensor import Tensor

@unittest.skipIf(not OPENCL_AVAILABLE, "OpenCL backend not available or enabled.")
class TestLayer3(unittest.TestCase):

    def test_opencl_backend_fusion(self):
        a_np = np.random.rand(10, 10).astype(np.float32)
        b_np = np.random.rand(10, 10).astype(np.float32)
        c_np = np.random.rand(10, 10).astype(np.float32)
        
        a = Tensor(a_np)
        b = Tensor(b_np)
        c = Tensor(c_np)

        # A fused chain of elementwise ops
        d = (a + b) * c

        # Realize the computation on the OpenCL device
        d_real_tensor = d.realize()
        d_ocl_data = d_real_tensor._op.inputs[0]

        # The ground truth computed with NumPy
        d_np = (a_np + b_np) * c_np

        np.testing.assert_allclose(d_ocl_data, d_np, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
