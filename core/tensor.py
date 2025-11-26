from typing import Tuple, Optional, List
import numpy as np
import math


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
        label: str = 'Tensor'
        ) -> None:
        self.data = np.array(data) # Create a numpy array as Tensor representation
        self.shape = self.data.shape # Get the shape from numpy .shape
        self.grad = np.zeros(self.shape) # By default gradients are zeros
        self._backward = lambda: None # The backward function to compute gradients
        self._prev = set(_children) # All other Tensors that lead to creation of this Tensor
        self._op = _op # The previous math operation that created this Tensor (None if it is an original Tensor)
        self.label = label # The name of a Tensor (debug)

# ======== Math operations ========
    # ADDITION
    def __add__(self, other):
        # Checks if other Tensor is numpy array or other data with the same shape
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f'Shapes have to be equal, but got: {self.shape} and {other.shape}'

        # Implement addition and return as new Tensor
        out = Tensor(
            data=self.data + other.data,
            _children=(self, other),
            _op='AddBackward'
        )

        # Derivative of addition operation propagates to all parent Tensors that form out Tensor
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    # SUBTRACTION
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # MULTIPLICATION
    def __mul__(self, other):
        # Checks if other Tensor is numpy array or other data with the same shape
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape == other.shape, f'Shapes have to be equal, but got 2 tensors: {self.shape} and {other.shape}'

        out = Tensor(
            data=self.data * other.data,
            _children=(self, other),
            _op='MulBackward'
        )

        # Derivative of muliplication
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # MATRIX MULTIPLICATION
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.shape[-1] == other.shape[0], f'Shapes have to be equal in N dim ((M, N) and (N, K)), got 2 tensors: {self.shape} and {other.shape}'

        out = Tensor(
            data=self.data @ other.data,
            _children=(self, other),
            _op='MatMulBackward'
        )

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __rmatmul__(self, other):
        return other @ self

    # DIVISION
    def __truediv__(self, other):
        return self * other**-1

    # EXPONENTIAL (self^other)
    def __pow__(self, other):
        # Checks if other is scaler
        assert isinstance(other, (int, float)), '__pow__ works only with scalers'

        out = Tensor(
            data=self.data ** other,
            _children=(self,),
            _op='PowBackward'
        )

        # Derivative of exponent
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward

        return out

    def __rpow__(self, other): # (other^self)
        # Checks if other is scaler
        assert isinstance(other, (int, float)), f'Base must to be a scaler'

        out = Tensor(
            data=other ** self.data,
            _children=(self,),
            _op='RevPowBackward'
        )

        def _backward():
            self.grad += (out.data * math.log(other)) * out.grad

        out._backward = _backward

        return out

    # NEGATION
    def __neg__(self):
        return self.data * -1

    # Logarithm (ln)
    def log(self):
        # Checks if Tensor has any elements that are less or equal to 0
        assert np.all(self.data > 0), f'Tensor values must be more than 0 to compute Log, but got: {self.data}'

        out = Tensor(
            data=np.log(self.data),
            _children=(self,),
            _op='LogBackward'
        )

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    # e EXPONENTIAL (e^self)
    def exp(self):
        out = Tensor(
            data=np.exp(self.data),
            _children=(self,),
            _op='ExpBackward'
        )

        def _backward():
            self.grad += self.data * out.grad

        out._backward = _backward

        return out

# ======== Representation of the object ========
    def __repr__(self) -> str:
        return f'{self.label}:\n{self.data}\nShape: {self.shape}'

    def backward(self):
        pass
