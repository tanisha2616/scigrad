"""
A NumPy-compatible API for scigrad, allowing it to be used as a drop-in
replacement for NumPy in many cases.
"""
from __future__ import annotations
import numpy as _np
from scigrad.tensor import Tensor

# --- Tensor Creation ---

def array(data, dtype=None) -> Tensor:
    """Creates a scigrad.Tensor from data."""
    # TODO: Handle dtype
    return Tensor(data)

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

# --- Reduction Operations ---

def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return a.sum(axis=axis, keepdims=keepdims)

# --- Utility functions to match NumPy API ---

def allclose(a: Tensor, b: Tensor, rtol=1e-05, atol=1e-08) -> bool:
    """
    Returns True if two tensors are element-wise equal within a tolerance.
    """
    a_np = a.realize()._op.inputs[0]
    b_np = b.realize()._op.inputs[0]
    return _np.allclose(a_np, b_np, rtol=rtol, atol=atol)
