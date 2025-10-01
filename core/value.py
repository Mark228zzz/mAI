import math
import random
from typing import Tuple, Optional, Callable
from contextlib import contextmanager


# Base Value class defines as a core operational element represented
# in mAI as scaler value using python variables. It is not efficient
# but for an example of a machine learning library like PyTorch.
_NO_GRAD = False


@contextmanager
def no_grad():
    global _NO_GRAD
    prev = _NO_GRAD
    _NO_GRAD = True
    try:
        yield
    finally:
        _NO_GRAD = prev


class Value:
    def __init__(
            self,
            data: float | int,
            _children: Optional[Tuple['Value', ...]] | None = (),
            _op: str = '',
            label: str = 'Value'
        ):

        self.data = data
        self.grad = 0.0 # By default grad is Zero
        self._backward = lambda: None # Backward function
        # All other Values that leads to creation of current Value object
        self._prev = set(_children)
        # The operation that leads to creation of current Value oject
        self._op = _op
        self.label = label # Debug label


    # ----- Representation -----
    def __repr__(self):
        return f'{self.label}(data={self.data}, grad={self.grad})'


    # ----- Numerical Operations -----
    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)

        # If gradient tracking is disabled, do not create graph connections
        children = () if _NO_GRAD else (self, other)
        out = Value(self.data + other.data, children, '+')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other): # self - other
        return self + (-other)

    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)

        children = () if _NO_GRAD else (self, other)
        out = Value(self.data * other.data, children, '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        # Only support scalar exponents (int, float)
        assert isinstance(other, (int, float)), "__pow__ only supports scalar exponents"
        children = () if _NO_GRAD else (self,)
        out = Value(self.data ** other, children, f'**{other}')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            # d/dx (x**k) = k * x**(k-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __rpow__(self, other):
    # handles (scalar ** Value)
        assert isinstance(other, (int, float)), "base must be a scalar"
        children = () if _NO_GRAD else (self,)
        out = Value(other ** self.data, children, f'{other}**')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            self.grad += (math.log(other) * out.data) * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def exp(self): # e**self
        x = self.data
        children = () if _NO_GRAD else (self,)
        out = Value(math.exp(x), children, 'exp')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        x = self.data
        children = () if _NO_GRAD else (self,)
        out = Value(math.log(x), children, 'log')

        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        return out

    # ----- Activation Functions -----
    def relu(self):
        children = () if _NO_GRAD else (self,)
        out = Value(self.data if self.data > 0 else 0, children, 'relu')

        def _backward():
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        # t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1) # ERROR OVER USE OF RECURSES
        t = math.tanh(x)
        children = () if _NO_GRAD else (self,)
        out = Value(t, children, 'tanh')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out


    # ----- Autograd Driver -----
    def backward(self):
        '''
        Make a topoligical order of all models parameters for backward pass
        in order to calculate all gradients right. Takes self as a root Value
        object where the backward pass begins and defines as 1.0.
        '''
        if _NO_GRAD:
            return
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Set the root Value as 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
