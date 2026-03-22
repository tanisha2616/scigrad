import unittest
import numpy as np
import os
from scigrad.scheduler import schedule

# Skip tests if OpenCL is not available or forced to CPU
try:
    import pyopencl as cl
    from scigrad.device import Device
    from scigrad.codegen.opencl import OpenCLBackend
    OPENCL_AVAILABLE = isinstance(Device, OpenCLBackend)
except Exception:
    OPENCL_AVAILABLE = False

from scigrad.tensor import Tensor


class TestSchedulerFusion(unittest.TestCase):

    def test_elementwise_chain_fuses_single_kernel(self):
        a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = Tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        c = Tensor(np.array([7.0, 8.0, 9.0], dtype=np.float32))

        out = (a + b) * c
        kernels = schedule(out._op)

        self.assertEqual(len(kernels), 1)
        self.assertEqual([u.op for u in kernels[0].uops], ["ADD", "MUL"])

    def test_reduction_terminates_fusion_group(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        c_np = np.array([7.0, 8.0, 9.0], dtype=np.float32)

        a = Tensor(a_np)
        b = Tensor(b_np)
        c = Tensor(c_np)

        out = ((a + b) * c).sum() + Tensor(np.array([1.0], dtype=np.float32))
        kernels = schedule(out._op)

        self.assertGreaterEqual(len(kernels), 2)
        self.assertEqual(kernels[0].uops[-1].op, "REDUCE_SUM")
        self.assertEqual(kernels[-1].uops[-1].op, "ADD")

        out_np = ((a_np + b_np) * c_np).sum() + np.array([1.0], dtype=np.float32)
        np.testing.assert_allclose(out.realize()._op.inputs[0], out_np, atol=1e-6, rtol=1e-6)

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
