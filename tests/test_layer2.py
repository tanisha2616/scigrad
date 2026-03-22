import unittest
import numpy as np
from scigrad.tensor import Tensor
from scigrad.scheduler import schedule

class TestLayer2(unittest.TestCase):

    def test_cpu_backend_simple_ops(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        a = Tensor(a_np)
        b = Tensor(b_np)

        # Test ADD
        c = a + b
        c_real_tensor = c.realize()
        c_real_data = c_real_tensor._op.inputs[0]
        np.testing.assert_allclose(c_real_data, a_np + b_np)

        # Test a chain of ops
        e = (a + b) * a
        e_real_tensor = e.realize()
        e_real_data = e_real_tensor._op.inputs[0]
        np.testing.assert_allclose(e_real_data, (a_np + b_np) * a_np)

    def test_scheduler_fusion(self):
        """Verify that the scheduler fuses three elementwise ops into one KernelSpec."""
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([4, 5, 6]))
        c = a + b
        d = c * a
        e = d + 1
        
        kernels = schedule(e._op)
        self.assertEqual(len(kernels), 1)
        kernel = kernels[0]
        self.assertEqual(len(kernel.uops), 4) # ADD, CONST, MUL, ADD

    def test_cpu_backend_execution(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        a = Tensor(a_np)
        b = Tensor(b_np)

        # Test ADD
        c = a + b
        c_real_tensor = c.realize()
        c_real_data = c_real_tensor._op.inputs[0]
        np.testing.assert_allclose(c_real_data, a_np + b_np)

        # Test a chain of ops
        e = (a + b) * a
        e_real_tensor = e.realize()
        e_real_data = e_real_tensor._op.inputs[0]
        np.testing.assert_allclose(e_real_data, (a_np + b_np) * a_np)

    def test_unary_ops(self):
        x_np = np.array([0.2, 0.5, 1.2], dtype=np.float32)
        x = Tensor(x_np)

        np.testing.assert_allclose(x.exp().realize()._op.inputs[0], np.exp(x_np), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.log().realize()._op.inputs[0], np.log(x_np), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.sqrt().realize()._op.inputs[0], np.sqrt(x_np), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.sin().realize()._op.inputs[0], np.sin(x_np), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.cos().realize()._op.inputs[0], np.cos(x_np), atol=1e-6, rtol=1e-6)

    def test_pad_and_shrink_ops(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        x = Tensor(x_np)

        pads = ((1, 1), (2, 0))
        x_pad = x.pad(pads)
        np.testing.assert_allclose(x_pad.realize()._op.inputs[0], np.pad(x_np, pads, mode='constant'), atol=1e-6, rtol=1e-6)

        slices = ((1, 3), (1, 3))
        x_shrink = x_pad.shrink(slices)
        np.testing.assert_allclose(x_shrink.realize()._op.inputs[0], np.pad(x_np, pads, mode='constant')[1:3, 1:3], atol=1e-6, rtol=1e-6)

    def test_permute_op(self):
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor(x_np)

        y = x.permute((1, 0, 2))
        np.testing.assert_allclose(y.realize()._op.inputs[0], np.transpose(x_np, (1, 0, 2)), atol=1e-6, rtol=1e-6)

    def test_tensor_api_parity_helpers(self):
        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x = Tensor(x_np)

        np.testing.assert_allclose(x.T.realize()._op.inputs[0], x_np.T, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.squeeze(0).realize()._op.inputs[0], x_np.squeeze(0), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.squeeze(0).unsqueeze(0).realize()._op.inputs[0], x_np, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(x.broadcast_to((2, 3)).realize()._op.inputs[0], np.broadcast_to(x_np, (2, 3)), atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
