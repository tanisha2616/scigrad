import unittest
import numpy as np

import scigrad.numpy as sgnp
from scigrad.tensor import Tensor
from scigrad.nn import (
    Conv2d,
    BatchNorm2d,
    LayerNorm,
    MultiheadAttention,
    Embedding,
    Dropout,
    Linear,
    optim,
)


class TinyNet:
    def __init__(self):
        self.fc = Linear(4, 2)

    def parameters(self):
        return [self.fc.weight, self.fc.bias]

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc(x)


class TinyTransformer:
    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int):
        self.in_proj = Linear(vocab_size, hidden_dim)
        self.attn1 = MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = LayerNorm(hidden_dim)
        self.attn2 = MultiheadAttention(hidden_dim, num_heads)
        self.norm2 = LayerNorm(hidden_dim)
        self.out_proj = Linear(hidden_dim, vocab_size)

    def __call__(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        h = self.norm1(self.attn1(h) + h)
        h = self.norm2(self.attn2(h) + h)
        return self.out_proj(h)

    def parameters(self):
        params = []
        for module in [self.in_proj, self.attn1, self.norm1, self.attn2, self.norm2, self.out_proj]:
            params.extend(module.parameters())
        return params


class TestLayer7(unittest.TestCase):

    def test_conv2d_shape(self):
        x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        conv = Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        y = conv(x)
        self.assertEqual(y.shape, (2, 4, 8, 8))

    def test_batchnorm2d_shape(self):
        x = Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
        bn = BatchNorm2d(4)
        y = bn(x)
        self.assertEqual(y.shape, (2, 4, 8, 8))

    def test_layernorm_shape(self):
        x = Tensor(np.random.randn(3, 5).astype(np.float32))
        ln = LayerNorm(5)
        y = ln(x)
        self.assertEqual(y.shape, (3, 5))

    def test_embedding_shape(self):
        idx = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
        emb = Embedding(20, 7)
        y = emb(idx)
        self.assertEqual(y.shape, (2, 3, 7))

    def test_dropout_train_eval(self):
        x = Tensor(np.ones((4, 4), dtype=np.float32))
        d = Dropout(0.5)

        d.train()
        y_train = d(x).realize()._op.inputs[0]
        self.assertEqual(y_train.shape, (4, 4))

        d.eval()
        y_eval = d(x).realize()._op.inputs[0]
        np.testing.assert_allclose(y_eval, np.ones((4, 4), dtype=np.float64), atol=1e-6, rtol=1e-6)

    def test_mha_shape(self):
        x = Tensor(np.random.randn(2, 5, 8).astype(np.float32))
        mha = MultiheadAttention(embed_dim=8, num_heads=2)
        y = mha(x)
        self.assertEqual(y.shape, (2, 5, 8))

    def test_sgd_optimizer_step(self):
        model = TinyNet()
        x = sgnp.array(np.random.randn(4, 4).astype(np.float32))
        y_target = sgnp.array(np.random.randn(4, 2).astype(np.float32))

        opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

        y_pred = model(x)
        loss = ((y_pred - y_target) * (y_pred - y_target)).sum()

        opt.zero_grad()
        loss.backward()

        before = model.fc.weight.realize()._op.inputs[0].copy()
        opt.step()
        after = model.fc.weight.realize()._op.inputs[0]

        self.assertFalse(np.allclose(before, after))

    def test_adam_optimizer_step(self):
        model = TinyNet()
        x = sgnp.array(np.random.randn(4, 4).astype(np.float32))
        y_target = sgnp.array(np.random.randn(4, 2).astype(np.float32))

        opt = optim.Adam(model.parameters(), lr=1e-2)

        y_pred = model(x)
        loss = ((y_pred - y_target) * (y_pred - y_target)).sum()

        opt.zero_grad()
        loss.backward()

        before = model.fc.bias.realize()._op.inputs[0].copy()
        opt.step()
        after = model.fc.bias.realize()._op.inputs[0]

        self.assertFalse(np.allclose(before, after))

    def test_tiny_transformer_loss_decreases(self):
        np.random.seed(0)

        vocab_size = 8
        hidden_dim = 64
        num_heads = 4
        batch_size = 8
        seq_len = 6

        # Periodic toy token stream.
        token_stream = np.arange(batch_size * (seq_len + 1)) % vocab_size
        token_stream = token_stream.reshape(batch_size, seq_len + 1)

        x_tokens = token_stream[:, :-1]
        y_tokens = token_stream[:, 1:]

        x_onehot = np.eye(vocab_size, dtype=np.float32)[x_tokens]
        y_onehot = np.eye(vocab_size, dtype=np.float32)[y_tokens]

        x = Tensor(x_onehot)
        y = Tensor(y_onehot)

        model = TinyTransformer(vocab_size=vocab_size, hidden_dim=hidden_dim, num_heads=num_heads)
        opt = optim.Adam(model.parameters(), lr=5e-3)

        first_loss = None
        last_loss = None

        for step in range(100):
            pred = model(x)
            diff = pred - y
            loss = (diff * diff).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_value = float(loss.realize()._op.inputs[0].reshape(-1)[0])
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value

        self.assertIsNotNone(first_loss)
        self.assertIsNotNone(last_loss)
        self.assertLess(last_loss, first_loss)


if __name__ == '__main__':
    unittest.main()
