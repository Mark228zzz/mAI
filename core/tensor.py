from typing import Tuple, Optional, List
import numpy as np


# Base Tensor class defines as a core operational element represented
# in mAI (Vectorized version) as a numpy arrays. It is not fulli efficient
# but recreation of a machine learning library like PyTorch or Tensorflow.
# Made for educational purpose to corely understand how ML libs work under the hood.
class Tensor:
    def __init__(
        self,
        data: float | List[float],
        _children: Optional[Tuple['Tensor', ...]] | None = (),
        _op: str | None = '',
        label: str = 'Value'
        ) -> None:
        self.data = np.array(data) # Create a numpy array as Tensor representation
        self.shape = self.data.shape # Get the shape from numpy .shape
        self.grad = np.zeros(self.shape) # By default gradients are zeros
        self._backward = lambda: None # The backward function to compute gradients
        self._prev = set(_children) # All other Tensors that lead to creation of this Tensor
        self._op = _op # The previous math operation that created this Tensor (None if it is an original Tensor)
        self.label = label # The name of a Tensor (debug)

# ======== Math operations ========
    def __add__(self, other):
        # Checks if other Tensor is numpy array or other data with the same shape
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f'Shapes have to be equal, but got: {self.shape} and {other.shape}'

        # Implement addition and return as new Tensor
        out: Tensor = Tensor(
            data=self.data + other.data,
            _children=(self, other),
            _op='AdditionBackward'
        )

        # The derivative of plus operation propagates to all Tensors
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

# ======== Representation of the object ========
    def __repr__(self) -> str:
        return f'{self.label}:\n{self.data}\nShape: {self.shape}'
