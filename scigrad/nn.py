"""
A simple neural network library built on top of scigrad.
"""
from typing import List
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
                    if isinstance(item, Module):
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
        return x @ self.weight + self.bias.expand((x.shape[0], self.bias.shape[1]))

    def __repr__(self):
        return f"Linear({self.weight.shape[0]}, {self.weight.shape[1]})"
