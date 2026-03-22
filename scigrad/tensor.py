from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Tuple, Any, List, Set, Optional, Callable
import numpy as np
import itertools

# --- Shape Inference ---
def broadcast_shape(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Infers the broadcasted shape of two input shapes."""
    s1_pad = (1,) * (len(s2) - len(s1)) + s1
    s2_pad = (1,) * (len(s1) - len(s2)) + s2
    out_shape = []
    for d1, d2 in zip(s1_pad, s2_pad):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Cannot broadcast shapes {s1} and {s2}")
        out_shape.append(max(d1, d2))
    return tuple(out_shape)

_uid_counter = itertools.count()

@dataclass(frozen=True)
class UOp:
    op: str
    inputs: Tuple[Union[UOp, np.ndarray], ...]
    args: Tuple[Any, ...] = ()
    uid: int = field(default_factory=lambda: next(_uid_counter), init=False)
    # These will be populated in __post_init__
    shape: Tuple[int, ...] = None
    dtype: str = None

    def __post_init__(self):
        if self.shape is not None and self.dtype is not None:
            return # Already initialized, likely from a copy

        # --- Shape & Dtype Inference ---
        op_shape, op_dtype = None, None
        if self.op in ('LOAD', 'CONST'):
            # For LOAD/CONST, the 'input' is the raw ndarray/value
            op_shape = self.inputs[0].shape
            op_dtype = str(self.inputs[0].dtype)
        elif self.op in ('NEG', 'EXP', 'LOG', 'SQRT', 'SIN', 'COS', 'ABS', 'SIGN', 'RECIP'):
            op_shape = self.inputs[0].shape
            op_dtype = self.inputs[0].dtype
        elif self.op == 'ASSIGN':
            dst_shape, src_shape = self.inputs[0].shape, self.inputs[1].shape
            if dst_shape != src_shape:
                raise ValueError(f"ASSIGN shape mismatch: dst={dst_shape}, src={src_shape}")
            op_shape = src_shape
            op_dtype = self.inputs[1].dtype
        elif self.op in ('ADD', 'MUL', 'MAX', 'CMPEQ', 'CMPLT'):
            op_shape = broadcast_shape(self.inputs[0].shape, self.inputs[1].shape)
            # Dtype promotion logic would go here
            op_dtype = self.inputs[0].dtype
        elif self.op == 'WHERE':
            # shape is broadcast of condition, x, y
            op_shape = broadcast_shape(self.inputs[0].shape, broadcast_shape(self.inputs[1].shape, self.inputs[2].shape))
            op_dtype = self.inputs[1].dtype # Dtype from data inputs
        elif self.op == 'RESHAPE':
            op_shape = self.args[0]
            op_dtype = self.inputs[0].dtype
        elif self.op == 'PAD':
            pads = self.args[0]
            op_shape = tuple(d + p0 + p1 for d, (p0, p1) in zip(self.inputs[0].shape, pads))
            op_dtype = self.inputs[0].dtype
        elif self.op == 'SHRINK':
            slices = self.args[0]
            op_shape = tuple(end - start for start, end in slices)
            op_dtype = self.inputs[0].dtype
        elif self.op == 'EXPAND':
            op_shape = self.args[0]
            op_dtype = self.inputs[0].dtype
        elif self.op in ('REDUCE_SUM', 'REDUCE_MAX', 'REDUCE_PROD'):
            axis = self.args[0]
            keepdims = self.args[1]
            ndim = len(self.inputs[0].shape)
            if axis is None:
                op_shape = (1,) * ndim if keepdims else (1,)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(sorted({a + ndim if a < 0 else a for a in axes}))
                if keepdims:
                    op_shape = tuple(1 if i in axes else d for i, d in enumerate(self.inputs[0].shape))
                else:
                    op_shape = tuple(d for i, d in enumerate(self.inputs[0].shape) if i not in axes)
            op_dtype = self.inputs[0].dtype
        elif self.op == 'CAST':
            op_shape = self.inputs[0].shape
            op_dtype = str(np.dtype(self.args[0]))
        elif self.op in ('TRANSPOSE', 'PERMUTE'):
            op_shape = tuple(self.inputs[0].shape[i] for i in self.args[0])
            op_dtype = self.inputs[0].dtype
        elif self.op == 'MATMUL':
            op_shape = self.inputs[0].shape[:-1] + self.inputs[1].shape[-1:]
            op_dtype = self.inputs[0].dtype
        else:
            raise NotImplementedError(f"Shape inference for op '{self.op}' not implemented.")

        # Object is frozen, so we have to use object.__setattr__
        object.__setattr__(self, 'shape', op_shape)
        object.__setattr__(self, 'dtype', op_dtype)

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        return self.uid == other.uid


class Tensor:
    def __init__(self, data: Union[np.ndarray, UOp, int, float], _children:Set[Tensor]=None, _op:str=None):
        self.grad: Optional[Tensor] = None
        self._backward: Callable = lambda: None
        self._prev: Set[Tensor] = _children or set()

        if isinstance(data, UOp):
            self._op = data
        elif isinstance(data, (int, float)):
            self._op = UOp(op='CONST', inputs=(np.array(data),))
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float64, copy=False)
            self._op = UOp(op='LOAD', inputs=(data,))

    @staticmethod
    def ones(*shape, **kwargs) -> Tensor:
        return Tensor(np.ones(shape, dtype=kwargs.get('dtype', np.float32)))

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        """Creates a tensor with values sampled from a uniform distribution."""
        dtype = kwargs.get('dtype', np.float32)
        return Tensor(np.random.uniform(low, high, size=shape).astype(dtype))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._op.shape

    @property
    def dtype(self) -> str:
        return self._op.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"<Tensor shape={self.shape} dtype='{self.dtype}' op={self._op.op}>"

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def assign(self, value: Union[Tensor, np.ndarray, int, float]) -> Tensor:
        value_t = value if isinstance(value, Tensor) else Tensor(value)
        out = Tensor(UOp(op='ASSIGN', inputs=(self._op, value_t._op)), _children={self, value_t})
        self._op = out._op
        return self

    # --- Unary Ops ---
    def __neg__(self) -> Tensor:
        out = Tensor(UOp(op='NEG', inputs=(self._op,)), _children={self})
        
        def _backward():
            if self.grad is not None:
                self.grad = self.grad - out.grad
            else:
                self.grad = -out.grad
        out._backward = _backward
        return out

    def recip(self) -> Tensor:
        out = Tensor(UOp(op='RECIP', inputs=(self._op,)), _children={self})
        
        def _backward():
            grad_for_self = out.grad * -(self.recip() * self.recip())
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(UOp(op='EXP', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = out.grad * out
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(UOp(op='LOG', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = out.grad * self.recip()
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def sqrt(self) -> Tensor:
        out = Tensor(UOp(op='SQRT', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = out.grad / (Tensor(2.0) * out)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def sin(self) -> Tensor:
        out = Tensor(UOp(op='SIN', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = out.grad * self.cos()
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def cos(self) -> Tensor:
        out = Tensor(UOp(op='COS', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = out.grad * -(self.sin())
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def abs(self) -> Tensor:
        out = Tensor(UOp(op='ABS', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = out.grad * self.sign()
            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

        out._backward = _backward
        return out

    def sign(self) -> Tensor:
        out = Tensor(UOp(op='SIGN', inputs=(self._op,)), _children={self})

        def _backward():
            grad_for_self = Tensor.zeros(*self.shape)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self

        out._backward = _backward
        return out

    def __abs__(self) -> Tensor:
        return self.abs()

    # --- Binary Ops ---
    def __add__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(UOp(op='ADD', inputs=(self._op, other._op)), _children={self, other})

        def _backward():
            if self.grad is not None:
                self.grad = self.grad + out.grad.sum_to_shape(self.shape)
            else:
                self.grad = out.grad.sum_to_shape(self.shape)
            
            if other.grad is not None:
                other.grad = other.grad + out.grad.sum_to_shape(other.shape)
            else:
                other.grad = out.grad.sum_to_shape(other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(UOp(op='MUL', inputs=(self._op, other._op)), _children={self, other})

        def _backward():
            grad_for_self = out.grad * other
            grad_for_other = out.grad * self

            if self.grad is not None:
                self.grad = self.grad + grad_for_self.sum_to_shape(self.shape)
            else:
                self.grad = grad_for_self.sum_to_shape(self.shape)

            if other.grad is not None:
                other.grad = other.grad + grad_for_other.sum_to_shape(other.shape)
            else:
                other.grad = grad_for_other.sum_to_shape(other.shape)

        out._backward = _backward
        return out

    def maximum(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(UOp(op='MAX', inputs=(self._op, other._op)), _children={self, other})

        def _backward():
            self_np = self.realize()._op.inputs[0]
            other_np = other.realize()._op.inputs[0]
            out_grad_np = out.grad.realize()._op.inputs[0]

            mask_self = (self_np >= other_np).astype(np.float64)
            mask_other = (other_np > self_np).astype(np.float64)

            grad_self = Tensor(out_grad_np * mask_self).sum_to_shape(self.shape)
            grad_other = Tensor(out_grad_np * mask_other).sum_to_shape(other.shape)

            if self.grad is not None:
                self.grad = self.grad + grad_self
            else:
                self.grad = grad_self

            if other.grad is not None:
                other.grad = other.grad + grad_other
            else:
                other.grad = grad_other

        out._backward = _backward
        return out

    def cmpeq(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(UOp(op='CMPEQ', inputs=(self._op, other._op)), _children={self, other})

        def _backward():
            if self.grad is None:
                self.grad = Tensor.zeros(*self.shape)
            if other.grad is None:
                other.grad = Tensor.zeros(*other.shape)

        out._backward = _backward
        return out

    def cmplt(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(UOp(op='CMPLT', inputs=(self._op, other._op)), _children={self, other})

        def _backward():
            if self.grad is None:
                self.grad = Tensor.zeros(*self.shape)
            if other.grad is None:
                other.grad = Tensor.zeros(*other.shape)

        out._backward = _backward
        return out

    def where(self, x, y) -> Tensor:
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = y if isinstance(y, Tensor) else Tensor(y)
        out = Tensor(UOp(op='WHERE', inputs=(self._op, x._op, y._op)), _children={self, x, y})

        def _backward():
            cond_np = self.realize()._op.inputs[0]
            grad_np = out.grad.realize()._op.inputs[0]

            cond_mask = (cond_np != 0).astype(np.float64)
            grad_x = Tensor(grad_np * cond_mask).sum_to_shape(x.shape)
            grad_y = Tensor(grad_np * (1.0 - cond_mask)).sum_to_shape(y.shape)

            if x.grad is not None:
                x.grad = x.grad + grad_x
            else:
                x.grad = grad_x

            if y.grad is not None:
                y.grad = y.grad + grad_y
            else:
                y.grad = grad_y

            if self.grad is None:
                self.grad = Tensor.zeros(*self.shape)

        out._backward = _backward
        return out

    def __sub__(self, other: Union[Tensor, int, float]) -> Tensor:
        return self + (-other)

    def __radd__(self, other): return self + other
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __lt__(self, other): return self.cmplt(other)

    def __truediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.recip()

    def __rtruediv__(self, other):
        return other * self.recip()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.slice(idx, idx + 1)
        raise NotImplementedError("Only integer indexing is currently supported.")

    def astype(self, dtype) -> Tensor:
        out = Tensor(UOp(op='CAST', inputs=(self._op,), args=(dtype,)), _children={self})

        def _backward():
            grad_for_self = out.grad.astype(self.dtype)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._op_factory('MATMUL', other)

        def _backward():
            # Grad for self (A): dout @ B^T
            # Grad for other (B): A^T @ dout
            # For batched matmul, transpose only the last two dimensions.
            def _swap_last_two(t: Tensor) -> Tensor:
                if t.ndim < 2:
                    return t
                axes = tuple(range(t.ndim - 2)) + (t.ndim - 1, t.ndim - 2)
                return t.permute(axes)

            grad_a = out.grad @ _swap_last_two(other)
            grad_b = _swap_last_two(self) @ out.grad
            
            if self.grad is not None:
                self.grad += grad_a.sum_to_shape(self.shape)
            else:
                self.grad = grad_a.sum_to_shape(self.shape)

            if other.grad is not None:
                other.grad += grad_b.sum_to_shape(other.shape)
            else:
                other.grad = grad_b.sum_to_shape(other.shape)
        
        out._backward = _backward
        return out

    def slice(self, start, stop):
        """Extracts a slice from the first dimension of the tensor."""
        # This is a simplification and not a full-featured slice op.
        # It's implemented via reshape and expand for demonstration.
        # A real slice op would be a primitive.
        
        # Create a one-hot-like mask
        mask_np = np.zeros(self.shape[0], dtype=np.float32)
        mask_np[start] = 1.0
        mask = Tensor(mask_np)
        
        # Reshape mask to be (shape[0], 1, 1, ...)
        mask_reshaped = mask.reshape(self.shape[0], *((1,) * (len(self.shape) - 1)))
        
        # Multiply self by the mask and sum along the first axis
        sliced = (self * mask_reshaped).sum(axis=0)
        return sliced

    @staticmethod
    def stack(tensors: List[Tensor]) -> Tensor:
        """
        Stacks a list of tensors along a new first dimension.
        This is a simplified implementation.
        """
        # Reshape each tensor to have a leading dimension of 1
        reshaped_tensors = [t.reshape(1, *t.shape) for t in tensors]
        
        # This is a placeholder for a real 'concat' operation
        # We will simulate it by adding to a zero tensor of the final shape
        final_shape = (len(tensors), *tensors[0].shape)
        result = Tensor.zeros(*final_shape)
        
        # This is a very inefficient way to concatenate
        for i, t_reshaped in enumerate(reshaped_tensors):
            # Create a zero tensor with the final shape
            temp = Tensor.zeros(*final_shape)
            
            # This is a hack to place the tensor at the right spot
            # A real implementation would need a 'pad' or 'scatter' primitive
            # For now, we expand and multiply by a one-hot mask
            mask_np = np.zeros(final_shape[0], dtype=np.float32)
            mask_np[i] = 1.0
            mask = Tensor(mask_np).reshape(final_shape[0], *((1,) * (len(final_shape) - 1)))
            
            result = result + (t_reshaped.expand(final_shape) * mask)
            
        return result

    def transpose(self, axes=None):
        # A simple transpose for 2D tensors for now
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))

        return self.permute(axes)

    def permute(self, axes: Tuple[int, ...]) -> Tensor:
        out = Tensor(UOp(op='PERMUTE', inputs=(self._op,), args=(axes,)), _children={self})

        inv_axes = tuple(np.argsort(axes))

        def _backward():
            if self.grad is not None:
                self.grad += out.grad.permute(inv_axes)
            else:
                self.grad = out.grad.permute(inv_axes)
        out._backward = _backward
        return out

    def _op_factory(self, op: str, *others: Tensor) -> Tensor:
        # A helper to create new tensors from operations
        _children = {self}.union(others)
        _inputs = (self._op,) + tuple(o._op for o in others)
        return Tensor(UOp(op=op, inputs=_inputs), _children=_children)

    # --- Shape Ops ---
    def reshape(self, *new_shape: int) -> Tensor:
        if isinstance(new_shape[0], tuple):
            new_shape = new_shape[0]
        if self.shape == new_shape: return self
        # Basic validation
        if np.prod(self.shape) != np.prod(new_shape):
            raise ValueError(f"Cannot reshape tensor of shape {self.shape} to {new_shape}")
        
        out = Tensor(UOp(op='RESHAPE', inputs=(self._op,), args=(new_shape,)), _children={self})
        
        def _backward():
            grad_for_self = out.grad.reshape(self.shape)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self
        out._backward = _backward
        
        return out

    def pad(self, pads: Tuple[Tuple[int, int], ...]) -> Tensor:
        if len(pads) != self.ndim:
            raise ValueError(f"pads length {len(pads)} must match ndim {self.ndim}")

        out = Tensor(UOp(op='PAD', inputs=(self._op,), args=(pads,)), _children={self})

        def _backward():
            shrink_slices = tuple((p0, p0 + d) for d, (p0, p1) in zip(self.shape, pads))
            grad_for_self = out.grad.shrink(shrink_slices)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self

        out._backward = _backward
        return out

    def shrink(self, slices: Tuple[Tuple[int, int], ...]) -> Tensor:
        if len(slices) != self.ndim:
            raise ValueError(f"slices length {len(slices)} must match ndim {self.ndim}")

        out = Tensor(UOp(op='SHRINK', inputs=(self._op,), args=(slices,)), _children={self})

        def _backward():
            pads = tuple((start, d - end) for d, (start, end) in zip(self.shape, slices))
            grad_for_self = out.grad.pad(pads)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self

        out._backward = _backward
        return out

    def expand(self, new_shape: Tuple[int, ...]) -> Tensor:
        if self.shape == new_shape: return self
        out = Tensor(UOp(op='EXPAND', inputs=(self._op,), args=(new_shape,)), _children={self})

        def _backward():
            grad_for_self = out.grad.sum_to_shape(self.shape)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self
        out._backward = _backward

        return out

    def broadcast_to(self, new_shape: Tuple[int, ...]) -> Tensor:
        return self.expand(new_shape)

    def unsqueeze(self, axis: int) -> Tensor:
        if axis < 0:
            axis += self.ndim + 1
        new_shape = self.shape[:axis] + (1,) + self.shape[axis:]
        return self.reshape(new_shape)

    def squeeze(self, axis: Optional[int] = None) -> Tensor:
        if axis is None:
            new_shape = tuple(d for d in self.shape if d != 1)
            if len(new_shape) == 0:
                new_shape = (1,)
            return self.reshape(new_shape)

        if axis < 0:
            axis += self.ndim
        if self.shape[axis] != 1:
            raise ValueError(f"Cannot squeeze axis {axis} with size {self.shape[axis]}")
        new_shape = self.shape[:axis] + self.shape[axis+1:]
        if len(new_shape) == 0:
            new_shape = (1,)
        return self.reshape(new_shape)

    # --- Reduction Ops ---
    def sum(self, axis=None, keepdims=False) -> Tensor:
        out = Tensor(UOp(op='REDUCE_SUM', inputs=(self._op,), args=(axis, keepdims)), _children={self})

        def _backward():
            # The gradient needs to be expanded back to the input's shape
            grad_for_self = out.grad.expand(self.shape)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self
        out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False) -> Tensor:
        out = Tensor(UOp(op='REDUCE_MAX', inputs=(self._op,), args=(axis, keepdims)), _children={self})

        def _backward():
            x_np = self.realize()._op.inputs[0]
            y_np = out.realize()._op.inputs[0]

            if axis is None:
                y_exp = np.broadcast_to(y_np.reshape((1,) * self.ndim if keepdims else (1,)), self.shape)
                mask = (x_np == y_exp).astype(np.float64)
                denom = np.maximum(mask.sum(), 1.0)
                grad_np = (out.grad.realize()._op.inputs[0].reshape((1,) * self.ndim if keepdims else (1,)) * mask) / denom
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                y_keep = y_np if keepdims else np.expand_dims(y_np, axis=axes)
                g_keep = out.grad.realize()._op.inputs[0] if keepdims else np.expand_dims(out.grad.realize()._op.inputs[0], axis=axes)
                y_exp = np.broadcast_to(y_keep, self.shape)
                mask = (x_np == y_exp).astype(np.float64)
                denom = np.maximum(mask.sum(axis=axes, keepdims=True), 1.0)
                grad_np = (np.broadcast_to(g_keep, self.shape) * mask) / denom

            grad_for_self = Tensor(grad_np)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self

        out._backward = _backward
        return out

    def prod(self, axis=None, keepdims=False) -> Tensor:
        out = Tensor(UOp(op='REDUCE_PROD', inputs=(self._op,), args=(axis, keepdims)), _children={self})

        def _backward():
            x_np = self.realize()._op.inputs[0]
            y_np = out.realize()._op.inputs[0]
            g_np = out.grad.realize()._op.inputs[0]

            if axis is None:
                y_keep = y_np.reshape((1,) * self.ndim if keepdims else (1,))
                g_keep = g_np.reshape((1,) * self.ndim if keepdims else (1,))
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                y_keep = y_np if keepdims else np.expand_dims(y_np, axis=axes)
                g_keep = g_np if keepdims else np.expand_dims(g_np, axis=axes)

            y_exp = np.broadcast_to(y_keep, self.shape)
            g_exp = np.broadcast_to(g_keep, self.shape)
            grad_np = g_exp * (y_exp / x_np)

            grad_for_self = Tensor(grad_np)
            if self.grad is not None:
                self.grad = self.grad + grad_for_self
            else:
                self.grad = grad_for_self

        out._backward = _backward
        return out

    def sum_to_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Reduces a tensor down to a target shape via summation."""
        if self.shape == shape:
            return self

        result = self

        # First reduce any leading dimensions that don't exist in target.
        if result.ndim > len(shape):
            leading_axes = tuple(range(result.ndim - len(shape)))
            result = result.sum(axis=leading_axes, keepdims=False)

        # Then reduce broadcasted axes where target expects dimension 1.
        axes_to_reduce = []
        for i, (d_self, d_target) in enumerate(zip(result.shape, shape)):
            if d_self == d_target:
                continue
            if d_target == 1:
                axes_to_reduce.append(i)
            else:
                raise ValueError(f"Cannot reduce shape {self.shape} to {shape}")

        if axes_to_reduce:
            result = result.sum(axis=tuple(axes_to_reduce), keepdims=True)

        if result.shape != shape:
            result = result.reshape(shape)

        return result

    @staticmethod
    def zeros(*shape, **kwargs):
        return Tensor(np.zeros(shape, dtype=kwargs.get('dtype', np.float32)))

    def realize(self) -> Tensor:
        """
        Triggers the computation of the tensor.
        This will schedule, codegen, and execute the graph.
        """
        from scigrad.device import Device
        # The result is a concrete numpy array
        realized_data = Device.realize(self._op)
        # We wrap it in a new Tensor that is a LOAD op,
        # so it can be used in subsequent computations.
        return Tensor(realized_data)

    def backward(self):
        # Build a reverse topological sort of the graph
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # Initialize gradients
        self.grad = Tensor.ones(*self.shape)
        for t in self._prev:
            if t.grad is None:
                t.grad = Tensor.zeros(*t.shape)

        for v in reversed(topo):
            v._backward()

