import numpy as np
import scigrad.numpy as sgnp
from scigrad.nn import Linear


def main():
    print("=== scigrad quickstart demo ===")

    # 1) Basic tensor math
    a = sgnp.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = sgnp.array([4.0, 5.0, 6.0], dtype=np.float32)
    c = sgnp.add(a, b)
    print("add:", c.realize()._op.inputs[0])

    # 2) Autodiff
    x = sgnp.array([3.0], dtype=np.float32)
    y = sgnp.array([4.0], dtype=np.float32)
    z = sgnp.add(sgnp.multiply(x, y), x)  # z = x*y + x
    z.backward()
    print("grad x (expected ~5):", x.grad.realize()._op.inputs[0])
    print("grad y (expected ~3):", y.grad.realize()._op.inputs[0])

    # 3) Tiny NN forward pass
    layer = Linear(3, 2)
    inp = sgnp.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = layer(inp)
    print("linear output shape:", out.shape)
    print("linear output values:", out.realize()._op.inputs[0])


if __name__ == "__main__":
    main()
