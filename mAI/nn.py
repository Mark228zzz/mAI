import math
import random
from typing import Tuple, Optional, Callable


# Base Value class defines as a core operational element represented
# in mAI as scaler value using python variables. It is not efficient
# but for an example of a machine learning library like PyTorch.
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
        return f'{self.label}(data={self.data})'


    # ----- Numerical Operations -----
    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

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

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        # Only support scalar exponents (int, float)
        assert isinstance(other, (int, float)), "__pow__ only supports scalar exponents"
        out = Value(self.data ** other, (self,), f'**{other}')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            # d/dx (x**k) = k * x**(k-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __rpow__(self, other):
    # handles (scalar ** Value)
        assert isinstance(other, (int, float)), "base must be a scalar"
        out = Value(other ** self.data, (self,), f'{other}**')

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
        out = Value(math.exp(x), (self, ), 'exp')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out


    # ----- Activation Functions -----
    def tanh(self):
        x = self.data
        # t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        t = math.tanh(x)
        out = Value(t, (self, ), 'tanh')

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


class Neuron:
    '''
    Create a Neuron representation. A neuron why has N inputs therefore N weights,
    one bias term inside and an activation function. Works like perceptron
    '''
    def __init__(self, nin, act_f: Callable = ...):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act_f = act_f

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = self.act_f(act)
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, activation_function: Callable):
        self.neurons = [Neuron(nin, activation_function) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts, activation_function: Callable = ...):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation_function) for i in range(len(nouts))]

    def __call__(self, xs):
        if isinstance(xs[0], (int, float, Value)):
            xs = [xs]
        out = []
        for x in xs:
            for layer in self.layers:
                x = layer(x)
            out.append(x)
        return out if len(out) > 1 else out[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
