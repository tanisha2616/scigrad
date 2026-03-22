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

if __name__ == '__main__':
    unittest.main()
