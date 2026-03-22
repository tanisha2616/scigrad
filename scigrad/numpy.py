"""
A NumPy-compatible API for scigrad, allowing it to be used as a drop-in
replacement for NumPy in many cases.
"""
from __future__ import annotations
import numpy as _np
from scigrad.tensor import Tensor

ndarray = Tensor


def _as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _as_numpy(x):
    if isinstance(x, Tensor):
        return x.realize()._op.inputs[0]
    return _np.array(x)

# --- Tensor Creation ---

def array(data, dtype=None) -> Tensor:
    """Creates a scigrad.Tensor from data."""
    arr = _np.array(data, dtype=dtype) if dtype is not None else _np.array(data)
    return Tensor(arr)

def ones(*shape, dtype=None) -> Tensor:
    """Creates a scigrad.Tensor of ones."""
    return Tensor.ones(*shape, dtype=dtype or _np.float32)

def zeros(*shape, dtype=None) -> Tensor:
    """Creates a scigrad.Tensor of zeros."""
    return Tensor.zeros(*shape, dtype=dtype or _np.float32)

# --- Element-wise Operations ---

def add(x1: Tensor, x2: Tensor) -> Tensor:
    return x1 + x2

def multiply(x1: Tensor, x2: Tensor) -> Tensor:
    return x1 * x2

def negative(x: Tensor) -> Tensor:
    return -x

def reciprocal(x: Tensor) -> Tensor:
    return x.recip()

def divide(x1: Tensor, x2: Tensor) -> Tensor:
    return x1 / x2

# --- Shape Manipulation ---

def reshape(a: Tensor, newshape: tuple[int, ...]) -> Tensor:
    return a.reshape(newshape)


def transpose(a: Tensor, axes=None) -> Tensor:
    return _as_tensor(a).transpose(axes)


def permute(a: Tensor, axes: tuple[int, ...]) -> Tensor:
    return _as_tensor(a).permute(axes)


def broadcast_to(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    return a.expand(shape)


def pad(a: Tensor, pad_width) -> Tensor:
    return _as_tensor(a).pad(tuple(tuple(p) for p in pad_width))

# --- Reduction Operations ---

def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return a.sum(axis=axis, keepdims=keepdims)


def max(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return _as_tensor(a).max(axis=axis, keepdims=keepdims)


def prod(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return _as_tensor(a).prod(axis=axis, keepdims=keepdims)


def astype(a: Tensor, dtype) -> Tensor:
    return _as_tensor(a).astype(dtype)


def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = _as_tensor(a)
    if axis is None:
        denom = _np.prod(a.shape)
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        denom = _np.prod([a.shape[i] for i in axes])
    return a.sum(axis=axis, keepdims=keepdims) / Tensor(float(denom))


def std(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = _as_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    centered = a - m
    var_t = mean(centered * centered, axis=axis, keepdims=keepdims)
    return Tensor(_np.sqrt(var_t.realize()._op.inputs[0]))


def var(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = _as_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    centered = a - m
    return mean(centered * centered, axis=axis, keepdims=keepdims)


def where(condition, x, y) -> Tensor:
    return _as_tensor(condition).where(_as_tensor(x), _as_tensor(y))


def maximum(x1, x2) -> Tensor:
    return _as_tensor(x1).maximum(_as_tensor(x2))


def equal(x1, x2) -> Tensor:
    return _as_tensor(x1).cmpeq(_as_tensor(x2))


def less(x1, x2) -> Tensor:
    return _as_tensor(x1).cmplt(_as_tensor(x2))


def clip(a, a_min, a_max) -> Tensor:
    return Tensor(_np.clip(_as_numpy(a), a_min, a_max))


def abs(x) -> Tensor:
    return _as_tensor(x).abs()


def sqrt(x) -> Tensor:
    return _as_tensor(x).sqrt()


def exp(x) -> Tensor:
    return _as_tensor(x).exp()


def log(x) -> Tensor:
    return _as_tensor(x).log()


def sin(x) -> Tensor:
    return _as_tensor(x).sin()


def cos(x) -> Tensor:
    return _as_tensor(x).cos()


def linspace(start, stop, num=50, endpoint=True, dtype=None) -> Tensor:
    return Tensor(_np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype))


def arange(start, stop=None, step=1, dtype=None) -> Tensor:
    if stop is None:
        return Tensor(_np.arange(start, dtype=dtype))
    return Tensor(_np.arange(start, stop, step, dtype=dtype))


def concatenate(arrays, axis=0) -> Tensor:
    return Tensor(_np.concatenate([_as_numpy(a) for a in arrays], axis=axis))


def stack(arrays, axis=0) -> Tensor:
    return Tensor(_np.stack([_as_numpy(a) for a in arrays], axis=axis))


def dot(a, b) -> Tensor:
    a = _as_tensor(a)
    b = _as_tensor(b)
    if a.ndim == 1 and b.ndim == 1:
        return (a * b).sum()
    return a @ b


def einsum(subscripts, *operands) -> Tensor:
    return Tensor(_np.einsum(subscripts, *[_as_numpy(o) for o in operands]))


def argmax(a, axis=None):
    return _np.argmax(_as_numpy(a), axis=axis)


def argmin(a, axis=None):
    return _np.argmin(_as_numpy(a), axis=axis)


def argsort(a, axis=-1):
    return _np.argsort(_as_numpy(a), axis=axis)


def searchsorted(a, v, side='left'):
    return _np.searchsorted(_as_numpy(a), _as_numpy(v), side=side)


def interp(x, xp, fp, left=None, right=None, period=None) -> Tensor:
    return Tensor(_np.interp(_as_numpy(x), _as_numpy(xp), _as_numpy(fp), left=left, right=right, period=period))


class fft:
    @staticmethod
    def fft(a):
        return Tensor(_np.fft.fft(_as_numpy(a)))

    @staticmethod
    def ifft(a):
        return Tensor(_np.fft.ifft(_as_numpy(a)))

    @staticmethod
    def fftfreq(n, d=1.0):
        return Tensor(_np.fft.fftfreq(n, d=d))

# --- Utility functions to match NumPy API ---

def allclose(a: Tensor, b: Tensor, rtol=1e-05, atol=1e-08) -> bool:
    """
    Returns True if two tensors are element-wise equal within a tolerance.
    """
    a_np = a.realize()._op.inputs[0]
    b_np = b.realize()._op.inputs[0]
    return _np.allclose(a_np, b_np, rtol=rtol, atol=atol)
