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
        elif self.op == 'EXPAND':
            op_shape = self.args[0]
            op_dtype = self.inputs[0].dtype
        elif self.op == 'REDUCE_SUM':
            axis = self.args[0]
            keepdims = self.args[1]
            if axis is None:
                op_shape = (1,) * len(self.inputs[0].shape) if keepdims else (1,)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                if keepdims:
                    op_shape = tuple(1 if i in axes else d for i, d in enumerate(self.inputs[0].shape))
                else:
                    op_shape = tuple(d for i, d in enumerate(self.inputs[0].shape) if i not in axes)
            op_dtype = self.inputs[0].dtype
        elif self.op == 'TRANSPOSE':
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

    def __sub__(self, other: Union[Tensor, int, float]) -> Tensor:
        return self + (-other)

    def __radd__(self, other): return self + other
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other

    def __truediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.recip()

    def __rtruediv__(self, other):
        return other * self.recip()

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._op_factory('MATMUL', other)

        def _backward():
            # Grad for self (A): dout @ B.T
            # Grad for other (B): A.T @ dout
            grad_a = out.grad @ other.transpose()
            grad_b = self.transpose() @ out.grad
            
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
        
        # For matmul grad, we need a primitive transpose
        # Let's assume we have a TRANSPOSE op
        out = Tensor(UOp(op='TRANSPOSE', inputs=(self._op,), args=(axes,)), _children={self})

        def _backward():
            # The gradient of a transpose is the transpose of the gradient
            if self.grad is not None:
                self.grad += out.grad.transpose(axes)
            else:
                self.grad = out.grad.transpose(axes)
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

    def sum_to_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Reduces a tensor down to a target shape via summation."""
        if self.shape == shape:
            return self
        
        # Align shapes by prepending 1s to the shorter shape
        aligned_self_shape = (1,) * (len(shape) - len(self.shape)) + self.shape
        
        # Figure out which axes to sum over
        sum_axes = [i for i, (d_self, d_target) in enumerate(zip(aligned_self_shape, shape)) if d_self != d_target]
        
        # Perform the sum
        summed = self.sum(axis=tuple(sum_axes), keepdims=True)
        
        # The result might have extra dims of size 1, so reshape to the final target shape
        return summed.reshape(shape)

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

