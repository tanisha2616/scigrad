import unittest
import numpy as np
import scigrad.numpy as sgnp

class TestLayer5(unittest.TestCase):

    def test_array_creation(self):
        np_a = np.array([1, 2, 3])
        sg_a = sgnp.array([1, 2, 3])
        self.assertTrue(sgnp.allclose(sg_a, sgnp.array(np_a)))

    def test_ones_zeros_creation(self):
        np_ones = np.ones((2, 3))
        sg_ones = sgnp.ones(2, 3)
        self.assertTrue(sgnp.allclose(sg_ones, sgnp.array(np_ones)))

        np_zeros = np.zeros((3, 2))
        sg_zeros = sgnp.zeros(3, 2)
        self.assertTrue(sgnp.allclose(sg_zeros, sgnp.array(np_zeros)))

    def test_elementwise_ops(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        np_b = np.array([4, 5, 6], dtype=np.float32)
        sg_a = sgnp.array(np_a)
        sg_b = sgnp.array(np_b)

        # Add
        np_c = np_a + np_b
        sg_c = sgnp.add(sg_a, sg_b)
        self.assertTrue(sgnp.allclose(sg_c, sgnp.array(np_c)))

        # Multiply
        np_d = np_a * np_b
        sg_d = sgnp.multiply(sg_a, sg_b)
        self.assertTrue(sgnp.allclose(sg_d, sgnp.array(np_d)))

        # Divide
        np_e = np_a / np_b
        sg_e = sgnp.divide(sg_a, sg_b)
        self.assertTrue(sgnp.allclose(sg_e, sgnp.array(np_e)))

        # Negative
        np_f = -np_a
        sg_f = sgnp.negative(sg_a)
        self.assertTrue(sgnp.allclose(sg_f, sgnp.array(np_f)))

    def test_reduction_ops(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        sg_a = sgnp.array(np_a)

        # Sum all
        np_b = np.sum(np_a)
        sg_b = sgnp.sum(sg_a)
        self.assertTrue(sgnp.allclose(sg_b, sgnp.array(np_b)))

        # Sum axis
        np_c = np.sum(np_a, axis=1)
        sg_c = sgnp.sum(sg_a, axis=1)
        self.assertTrue(sgnp.allclose(sg_c, sgnp.array(np_c)))

    def test_autodiff_with_numpy_api(self):
        # Test that autodiff still works when using the numpy api
        a_np = np.array([3.0], dtype=np.float32)
        b_np = np.array([4.0], dtype=np.float32)
        
        a = sgnp.array(a_np)
        b = sgnp.array(b_np)

        c = sgnp.multiply(a, b)
        d = sgnp.add(c, a)
        d.backward()

        # dc/da = b = 4
        # dd/dc = 1
        # dd/da = 1
        # dd/da_total = (dd/dc * dc/da) + dd/da = b + 1 = 5
        expected_grad_a = b_np + 1
        self.assertTrue(sgnp.allclose(a.grad, sgnp.array(expected_grad_a)))

if __name__ == '__main__':
    unittest.main()
