import unittest
import numpy as np
from scigrad.tensor import Tensor

def numerical_grad(f, *inputs, eps=1e-4):
    """
    Computes numerical gradient of f with respect to each input using finite differences.
    This version is designed to be completely stateless to avoid graph interference.
    """
    # Store the raw numpy data of the original inputs
    input_data = [t.realize()._op.inputs[0].copy() for t in inputs]
    grads = []

    # Iterate through each input tensor to compute the gradient for it
    for i in range(len(input_data)):
        grad = np.zeros_like(input_data[i])
        it = np.nditer(input_data[i], flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            old_val = input_data[i][idx]

            # --- Calculate f(x + eps) ---
            # Create a modified copy of the specific input array
            x_plus_np = input_data[i].copy()
            x_plus_np[idx] = old_val + eps
            # Create a fresh list of Tensors for this specific calculation
            inputs_plus = [Tensor(d.copy()) for d in input_data]
            inputs_plus[i] = Tensor(x_plus_np) # Substitute the modified tensor
            out_plus = f(*inputs_plus).realize()._op.inputs[0]

            # --- Calculate f(x - eps) ---
            # Create another modified copy
            x_minus_np = input_data[i].copy()
            x_minus_np[idx] = old_val - eps
            # Create another fresh list of Tensors
            inputs_minus = [Tensor(d.copy()) for d in input_data]
            inputs_minus[i] = Tensor(x_minus_np) # Substitute the modified tensor
            out_minus = f(*inputs_minus).realize()._op.inputs[0]

            # --- Central Difference ---
            grad[idx] = np.sum(out_plus - out_minus) / (2 * eps)

            it.iternext()
        grads.append(grad)
        
    return grads if len(grads) > 1 else grads[0]


class TestLayer4(unittest.TestCase):

    def test_add_grad(self):
        a = Tensor(np.array([2.0, 3.0], dtype=np.float32))
        b = Tensor(np.array([5.0, 7.0], dtype=np.float32))
        
        c = (a + b).sum()
        c.backward()

        num_grad_a, num_grad_b = numerical_grad(lambda x, y: (x + y).sum(), a, b)

        np.testing.assert_allclose(a.grad.realize()._op.inputs[0], num_grad_a, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b.grad.realize()._op.inputs[0], num_grad_b, rtol=1e-4, atol=1e-5)

    def test_mul_grad(self):
        a = Tensor(np.array([2.0, 3.0], dtype=np.float32))
        b = Tensor(np.array([5.0, 7.0], dtype=np.float32))
        
        c = (a * b).sum()
        c.backward()

        num_grad_a, num_grad_b = numerical_grad(lambda x, y: (x * y).sum(), a, b)

        np.testing.assert_allclose(a.grad.realize()._op.inputs[0], num_grad_a, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b.grad.realize()._op.inputs[0], num_grad_b, rtol=1e-4, atol=1e-5)

    def test_chain_rule_grad(self):
        a = Tensor(np.array([2.0], dtype=np.float32))
        b = Tensor(np.array([5.0], dtype=np.float32))
        
        c = (a * b) + a
        c.backward()

        num_grad_a = numerical_grad(lambda x: (x * b) + x, a)
        
        # Analytical grad: dc/da = b + 1
        analytical_grad_a = b.realize()._op.inputs[0] + 1
        
        grad_a_real = a.grad.realize()._op.inputs[0]
        np.testing.assert_allclose(grad_a_real, num_grad_a, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(grad_a_real, analytical_grad_a, rtol=1e-4, atol=1e-5)

    def test_div_grad(self):
        a = Tensor(np.array([10.0], dtype=np.float32))
        b = Tensor(np.array([2.0], dtype=np.float32))

        c = (a / b).sum()
        c.backward()

        num_grad_a, num_grad_b = numerical_grad(lambda x, y: (x / y).sum(), a, b)

        np.testing.assert_allclose(a.grad.realize()._op.inputs[0], num_grad_a, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b.grad.realize()._op.inputs[0], num_grad_b, rtol=1e-4, atol=1e-5)

    def test_max_grad_unique(self):
        a = Tensor(np.array([1.0, 3.0, 2.0], dtype=np.float32))
        y = a.max()
        y.backward()

        expected = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(a.grad.realize()._op.inputs[0], expected, rtol=1e-6, atol=1e-6)

    def test_prod_grad(self):
        a = Tensor(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        y = a.prod()
        y.backward()

        num_grad_a = numerical_grad(lambda x: x.prod(), a)
        np.testing.assert_allclose(a.grad.realize()._op.inputs[0], num_grad_a, rtol=1e-3, atol=1e-4)

    def test_where_grad(self):
        cond = Tensor(np.array([1.0, 0.0, 1.0], dtype=np.float32))
        x = Tensor(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        y = Tensor(np.array([5.0, 6.0, 7.0], dtype=np.float32))

        out = cond.where(x, y).sum()
        out.backward()

        np.testing.assert_allclose(x.grad.realize()._op.inputs[0], np.array([1.0, 0.0, 1.0]), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(y.grad.realize()._op.inputs[0], np.array([0.0, 1.0, 0.0]), atol=1e-6, rtol=1e-6)

    def test_permute_grad(self):
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor(x_np)

        out = x.permute((1, 0, 2)).sum()
        out.backward()

        np.testing.assert_allclose(x.grad.realize()._op.inputs[0], np.ones_like(x_np, dtype=np.float64), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
