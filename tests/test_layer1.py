import unittest
import numpy as np
from scigrad.tensor import Tensor, UOp

class TestTensorUOp(unittest.TestCase):

    def test_lazy_graph_construction(self):
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([4, 5, 6]))
        c = Tensor(np.array([7, 8, 9]))

        d = a + b
        e = d * c
        f = -e
        g = f.reshape((1, 3))
        h = g + 1 # Test broadcasting with a scalar

        # 1. Check that no computation has happened. The root op is a UOp.
        self.assertIsInstance(h._op, UOp)

        # 2. Verify the graph structure and op names.
        # h = g + 1 -> ADD
        self.assertEqual(h._op.op, 'ADD')
        # g = f.reshape((1, 3)) -> RESHAPE
        g_op = h._op.inputs[0]
        self.assertEqual(g_op.op, 'RESHAPE')
        # f = -e -> NEG
        f_op = g_op.inputs[0]
        self.assertEqual(f_op.op, 'NEG')
        # e = d * c -> MUL
        e_op = f_op.inputs[0]
        self.assertEqual(e_op.op, 'MUL')
        # d = a + b -> ADD
        d_op = e_op.inputs[0]
        self.assertEqual(d_op.op, 'ADD')

        # 3. Check that the leaves are LOAD ops from the original numpy arrays
        a_op = d_op.inputs[0]
        b_op = d_op.inputs[1]
        c_op = e_op.inputs[1]
        self.assertEqual(a_op.op, 'LOAD')
        self.assertEqual(b_op.op, 'LOAD')
        self.assertEqual(c_op.op, 'LOAD')
        
        # Check the scalar in the final op
        scalar_op = h._op.inputs[1]
        self.assertEqual(scalar_op.op, 'CONST')


    def test_shape_inference(self):
        # Reshape
        a = Tensor(np.zeros((2, 3)))
        b = a.reshape((3, 2))
        self.assertEqual(b.shape, (3, 2))

        with self.assertRaises(ValueError):
            a.reshape((4, 2)) # Mismatched size

        # Broadcasting
        x = Tensor(np.zeros((1, 3)))
        y = Tensor(np.zeros((2, 1)))
        z = x + y
        self.assertEqual(z.shape, (2, 3))

        # Invalid broadcast
        with self.assertRaises(ValueError):
            w = Tensor(np.zeros((2, 3)))
            v = Tensor(np.zeros((2, 4)))
            u = w + v

if __name__ == '__main__':
    unittest.main()
