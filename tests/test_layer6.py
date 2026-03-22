import unittest
import scigrad.numpy as sgnp
from scigrad.nn import Module, Linear
from scigrad.tensor import Tensor

class SimpleNet(Module):
    """A simple two-layer neural network."""
    def __init__(self):
        super().__init__()
        self.l1 = Linear(10, 5)
        self.l2 = Linear(5, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        # No activation function for simplicity
        x = self.l2(x)
        return x

class TestLayer6(unittest.TestCase):

    def test_parameter_collection(self):
        model = SimpleNet()
        params = model.parameters()
        # l1.weight, l1.bias, l2.weight, l2.bias
        self.assertEqual(len(params), 4)
        self.assertIsInstance(params[0], Tensor)

    def test_training_step(self):
        """
        Tests if a single training step reduces the loss, which implies
        that gradients are being calculated and parameters are being updated.
        """
        # A simple dataset
        X_train = sgnp.array([[i/10.0] * 10 for i in range(5)], dtype='float32') # 5 samples, 10 features
        Y_train = sgnp.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype='float32') # 5 targets

        model = SimpleNet()
        
        # --- Get initial loss ---
        Y_pred_initial = model(X_train)
        # Mean Squared Error Loss
        loss_initial = ((Y_pred_initial - Y_train) * (Y_pred_initial - Y_train)).sum()
        loss_initial_val = loss_initial.realize()._op.inputs[0]

        # --- Perform a single training step ---
        
        # 1. Zero gradients
        for p in model.parameters():
            p.grad = None
            
        # 2. Backward pass
        loss_initial.backward()

        # 3. Update weights (simple SGD)
        lr = 0.01
        for p in model.parameters():
            # The parameter update is also part of the graph, but we don't
            # want to track gradients for it. We can create new Tensors
            # from the realized data to detach them from the graph.
            p_data = p.realize()._op.inputs[0]
            grad_data = p.grad.realize()._op.inputs[0]
            p_data = p_data - lr * grad_data
            
            # Replace the old parameter tensor with the new one
            p._op = Tensor(p_data)._op

        # --- Get new loss ---
        Y_pred_new = model(X_train)
        loss_new = ((Y_pred_new - Y_train) * (Y_pred_new - Y_train)).sum()
        loss_new_val = loss_new.realize()._op.inputs[0]

        # --- Assert that loss has decreased ---
        self.assertLess(loss_new_val, loss_initial_val)


if __name__ == '__main__':
    unittest.main()
