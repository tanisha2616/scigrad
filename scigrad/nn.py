"""Neural network layers and optimizers built on top of scigrad."""
from __future__ import annotations

from typing import List
import numpy as np
import scigrad.numpy as sgnp
from scigrad.tensor import Tensor

class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Returns a list of all learnable parameters in the module."""
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Tensor):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def train(self):
        """Sets the module in training mode."""
        self.training = True
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train()

    def eval(self):
        """Sets the module in evaluation mode."""
        self.training = False
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                value.eval()


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Xavier initialization for weights
        limit = (6 / (in_features + out_features))**0.5
        self.weight = Tensor.uniform(in_features, out_features, low=-limit, high=limit)
        self.bias = Tensor.uniform(1, out_features, low=-limit, high=limit)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        target_shape = x.shape[:-1] + (self.bias.shape[1],)
        bias_view_shape = (1,) * (len(target_shape) - 1) + (self.bias.shape[1],)
        bias = self.bias.reshape(bias_view_shape).expand(target_shape)
        return out + bias

    def __repr__(self):
        return f"Linear({self.weight.shape[0]}, {self.weight.shape[1]})"


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability p must be in [0, 1).")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float64) / keep_prob
        return x * Tensor(mask)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = Tensor.uniform(num_embeddings, embedding_dim, low=-0.1, high=0.1)

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices.realize()._op.inputs[0].astype(np.int64)
        w = self.weight.realize()._op.inputs[0]
        return Tensor(w[idx])


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones((1, normalized_shape), dtype=np.float64))
        self.bias = Tensor(np.zeros((1, normalized_shape), dtype=np.float64))

    def forward(self, x: Tensor) -> Tensor:
        axis = x.ndim - 1
        mean = x.sum(axis=axis, keepdims=True) / Tensor(float(x.shape[axis]))
        centered = x - mean
        var = (centered * centered).sum(axis=axis, keepdims=True) / Tensor(float(x.shape[axis]))
        normed = centered / (var + Tensor(self.eps)).sqrt()
        return normed * self.weight.expand(normed.shape) + self.bias.expand(normed.shape)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(np.ones((1, num_features, 1, 1), dtype=np.float64))
        self.bias = Tensor(np.zeros((1, num_features, 1, 1), dtype=np.float64))
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float64)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float64)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x_np = x.realize()._op.inputs[0]
            mean_np = x_np.mean(axis=(0, 2, 3), keepdims=True)
            var_np = x_np.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean_np
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var_np
            mean = Tensor(mean_np)
            var = Tensor(var_np)
        else:
            mean = Tensor(self.running_mean)
            var = Tensor(self.running_var)

        normed = (x - mean.expand(x.shape)) / (var.expand(x.shape) + Tensor(self.eps)).sqrt()
        return normed * self.weight.expand(x.shape) + self.bias.expand(x.shape)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        limit = (6 / (in_channels * kernel_size * kernel_size + out_channels)) ** 0.5
        self.weight = Tensor.uniform(out_channels, in_channels, kernel_size, kernel_size, low=-limit, high=limit)
        self.bias = Tensor.uniform(out_channels, low=-limit, high=limit)

    def forward(self, x: Tensor) -> Tensor:
        x_np = x.realize()._op.inputs[0]
        w_np = self.weight.realize()._op.inputs[0]
        b_np = self.bias.realize()._op.inputs[0]

        n, c_in, h, w = x_np.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        x_pad = np.pad(x_np, ((0, 0), (0, 0), (p, p), (p, p)))
        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1
        out = np.zeros((n, self.out_channels, h_out, w_out), dtype=np.float64)

        for i in range(h_out):
            for j in range(w_out):
                patch = x_pad[:, :, i * s:i * s + k, j * s:j * s + k]
                out[:, :, i, j] = np.tensordot(patch, w_np, axes=([1, 2, 3], [1, 2, 3])) + b_np

        return Tensor(out)


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, e = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).permute((0, 2, 1, 3))

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.permute((0, 2, 1, 3)).reshape(b, t, h * d)

    def forward(self, x: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        k_t = k.permute((0, 1, 3, 2))
        scores = (q @ k_t) / Tensor(float(self.head_dim) ** 0.5)

        # Differentiable softmax over last axis.
        scores_shifted = scores - scores.max(axis=-1, keepdims=True)
        attn = scores_shifted.exp()
        attn = attn / attn.sum(axis=-1, keepdims=True)

        ctx = attn @ v
        out = self._merge_heads(ctx)
        return self.out_proj(out)


class _Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(_Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 1e-3, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.realize()._op.inputs[0]) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.realize()._op.inputs[0]
            self._velocity[i] = self.momentum * self._velocity[i] + g
            p.assign(Tensor(p.realize()._op.inputs[0] - self.lr * self._velocity[i]))


class Adam(_Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.realize()._op.inputs[0]) for p in params]
        self.v = [np.zeros_like(p.realize()._op.inputs[0]) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.realize()._op.inputs[0]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.assign(Tensor(p.realize()._op.inputs[0] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)))


class AdamW(Adam):
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.realize()._op.inputs[0]
            w = p.realize()._op.inputs[0]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * w
            p.assign(Tensor(w - self.lr * update))


class optim:
    SGD = SGD
    Adam = Adam
    AdamW = AdamW
