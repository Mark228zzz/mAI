from typing import Tuple, Optional, List, Literal
import numpy as np


# Base Tensor class defines as a core operational element represented
# in mAI (Vectorized version) as a numpy arrays. It is not fulli efficient
# but recreation of a machine learning library like PyTorch or Tensorflow.
# Made for educational purpose to corely understand how ML libs work under the hood.
class Tensor:
    __slots__ = ['data', 'shape', 'require_grad', '_backward', '_prev', '_op', 'label', 'grad']

    def __init__(
        self,
        data: float | List[float],
        require_grad: bool = False,
        _children: Optional[Tuple['Tensor', ...]] | None = (),
        _op: str | None = '',
        label: str = 'Tensor'
        ) -> None:

        if isinstance(data, Tensor):
            data = data.data

        self.data = np.array(data, dtype=float) # Create a numpy array as Tensor representation
        self.shape = self.data.shape # Get the shape from numpy .shape
        self.require_grad = require_grad # If False no gradients will be created and calculated
        self.grad = np.zeros_like(self.data) if require_grad else None # By default gradients are ones (if require_grad is True)
        self._backward: callable = lambda: None # The backward function to compute gradients
        self._prev = set(_children) # All other Tensors that lead to creation of this Tensor
        self._op = _op # The previous math operation that created this Tensor (None if it is an original Tensor)
        self.label = label # The name of a Tensor (debug)

# ======== Math operations ========
    # ADDITION
    def __add__(self, other):
        # Checks if other Tensor is numpy array or other data with the same shape
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Implement addition and return as new Tensor
        out = Tensor(
            data=self.data + other.data,
            _children=(self, other),
            _op='AddBackward',
            require_grad=self.require_grad or other.require_grad
        )

        # Derivative of addition operation propagates to all parent Tensors that form out Tensor
        def _backward():
            if self.grad is not None:
                grad_self = out.grad
                ndims_added = len(out.shape) - len(self.shape)
                for i in range(ndims_added):
                    grad_self = grad_self.sum(axis=0)
                for i, (dim_self, dim_out) in enumerate(zip(self.shape, grad_self.shape)):
                    if dim_self == 1 and dim_out > 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad += grad_self

            if other.grad is not None:
                grad_other = out.grad
                ndims_added = len(out.shape) - len(other.shape)
                for i in range(ndims_added):
                    grad_other = grad_other.sum(axis=0)
                for i, (dim_other, dim_out) in enumerate(zip(other.shape, grad_other.shape)):
                    if dim_other == 1 and dim_out > 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other

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

        out = Tensor(
            data=self.data * other.data,
            _children=(self, other),
            _op='MulBackward',
            require_grad=self.require_grad or other.require_grad
        )

        # Derivative of muliplication
        def _backward():
            if self.grad is not None:
                grad_self = other.data * out.grad
                ndims_added = len(out.shape) - len(self.shape)
                for i in range(ndims_added):
                    grad_self = grad_self.sum(axis=0)
                for i, (dim_self, dim_out) in enumerate(zip(self.shape, grad_self.shape)):
                    if dim_self == 1 and dim_out > 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad += grad_self

            if other.grad is not None:
                grad_other = self.data * out.grad
                ndims_added = len(out.shape) - len(other.shape)
                for i in range(ndims_added):
                    grad_other = grad_other.sum(axis=0)
                for i, (dim_other, dim_out) in enumerate(zip(other.shape, grad_other.shape)):
                    if dim_other == 1 and dim_out > 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # MATRIX MULTIPLICATION
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data=self.data @ other.data,
            _children=(self, other),
            _op='MatMulBackward',
            require_grad=self.require_grad or other.require_grad
        )

        def _backward():
            if self.grad is not None: self.grad += out.grad @ other.data.T
            if other.grad is not None: other.grad += self.data.T @ out.grad

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
            _op='PowBackward',
            require_grad=self.require_grad
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
            _op='RevPowBackward',
            require_grad=self.require_grad
        )

        def _backward():
            self.grad += (out.data * other.log()) * out.grad

        out._backward = _backward

        return out

    # NEGATION
    def __neg__(self):
        return self * -1

    # Logarithm (ln)
    def log(self):
        out = Tensor(
            data=np.log(self.data),
            _children=(self,),
            _op='LogBackward',
            require_grad=self.require_grad
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
            _op='ExpBackward',
            require_grad=self.require_grad
        )

        def _backward():
            self.grad += self.data * out.grad

        out._backward = _backward

        return out

# ======== Activation Functions ========
    def relu(self):
        out = Tensor(
            data=np.maximum(self.data, 0),
            _children=(self,),
            _op='ReLUBackward',
            require_grad=self.require_grad
        )

        # Derivative of ReLU, when element more than 0 its gradient is 1 otherwise gradient is 0
        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        out = Tensor(
            data=np.tanh(self.data),
            _children=(self,),
            _op='TanhBackward',
            require_grad=self.require_grad
        )

        # Derivative of Tanh
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        out = Tensor(
            data=((1 + ((-self).exp())) ** -1).data,
            _children=(self,),
            _op='SigmoidBackward',
            require_grad=self.require_grad
        )

        # Derivative of Sigmoid
        def _backward():
            e_neg = (-self).exp()
            self.grad += (e_neg.data * (1 + e_neg.data) ** -2) * out.grad

        out._backward = _backward

        return out

    def leaky_relu(self, alpha: float = 0.01):
        out = Tensor(
            data=np.maximum(self.data, alpha * self.data),
            _children=(self,),
            _op='LeakyReLUBackward',
            require_grad=self.require_grad
        )

        def _backward():
            grad_input = np.ones_like(out.grad)
            grad_input[self.data <= 0] = alpha
            self.grad += grad_input * out.grad

        out._backward = _backward

        return out

# ======== Calculate gradients backwards ========
    def backward(self):
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
                topo.append(tensor)

        build_topo(self)

        # Set current Tensor's grads to ones, allowing backpropagation start
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.require_grad and node.grad is not None: node._backward()

# ======== Representation of the object ========
    def __repr__(self) -> str:
        repr = f'{self.label}:\n{self.data}\nShape: {self.shape}'
        return repr if not self.require_grad else repr + f'\nRequire Grad: {self.require_grad}'

# ======== NumPy functionality ========
    def reshape(self, shape: Tuple[int, ...], order: Literal['C', 'A', 'F'] = 'C') -> None:
        self.data = self.data.reshape(shape, order=order)
        if self.grad is not None: self.grad = self.grad.reshape(shape, order=order)

    def flatten(self, order: Literal['C', 'A', 'F', 'K'] = 'C') -> None:
        self.data = self.data.flatten(order=order)
        if self.grad is not None: self.grad = self.grad.flatten(order=order)

    def sum(self, axis: Tuple[int, ...] | None = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(
            data=self.data.sum(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op='SumBackward',
            require_grad=self.require_grad
        )

        def _backward():
            # Gradient broadcasts back to original shape
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                # Expand dims to match original shape for broadcasting
                grad_expanded = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                self.grad += np.broadcast_to(grad_expanded, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Tuple[int, ...] | None = None, keepdims: bool = False) -> 'Tensor':
        n = self.data.size if axis is None else np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

        out = Tensor(
            data=self.data.mean(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op='MeanBackward',
            require_grad=bool(self.require_grad)
        )
        def _backward():
            # Mean distributes gradient equally: d(mean)/dx = 1/n
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad / n
            else:
                grad_expanded = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                self.grad += np.broadcast_to(grad_expanded, self.data.shape) / n

        out._backward = _backward
        return out

    def max(self, axis: Tuple[int, ...] | None = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(
            data=self.data.max(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op='MaxBackward'
        )

        def _backward():
            # Gradient flows only to the maximum value(s)
            if axis is None:
                mask = (self.data == out.data)
            else:
                # Expand dims to match original shape
                max_expanded = np.expand_dims(out.data, axis=axis) if not keepdims else out.data
                mask = (self.data == max_expanded)

            # Distribute gradient to all max values (handles ties)
            grad_expanded = np.expand_dims(out.grad, axis=axis) if (axis is not None and not keepdims) else out.grad
            self.grad += mask * np.broadcast_to(grad_expanded, self.data.shape)

        out._backward = _backward
        return out

    def min(self, axis: Tuple[int, ...] | None = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(
            data=self.data.min(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op='MinBackward'
        )

        def _backward():
            # Gradient flows only to the minimum value(s)
            if axis is None:
                mask = (self.data == out.data)
            else:
                min_expanded = np.expand_dims(out.data, axis=axis) if not keepdims else out.data
                mask = (self.data == min_expanded)

            grad_expanded = np.expand_dims(out.grad, axis=axis) if (axis is not None and not keepdims) else out.grad
            self.grad += mask * np.broadcast_to(grad_expanded, self.data.shape)

        out._backward = _backward
        return out
