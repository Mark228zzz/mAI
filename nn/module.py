from typing import Callable, Iterable
import random
from ..core.value import Value
from ..core.parameter import Parameter
from ..core.module import Module

def _ensure_value_list(x: Iterable[float | int | Value]) -> list[Value]:
    vals = []
    for xi in x:
        vals.append(xi if isinstance(xi, Value) else Value(xi))
    return vals


class Neuron(Module):
    '''
    Create a Neuron representation. A neuron why has N inputs therefore N weights,
    one bias term inside and an activation function. Works like a simple perceptron.
    '''
    def __init__(self, nin, act_f: Callable = ...):
        self.w = [Parameter(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Parameter(random.uniform(-1, 1))
        self.act_f = act_f

    def forward(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = self.act_f(act)
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation_function: Callable):
        self.neurons = [Neuron(nin, activation_function) for _ in range(nout)]

    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activation_function: Callable = ...):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation_function) for i in range(len(nouts))]

    def forward(self, xs):
        is_batch = isinstance(xs[0], (list, tuple, Value))
        if not is_batch:
            xs = [xs]

        out = []
        for x in xs:
            for layer in self.layers:
                x = layer(x)
            if isinstance(x, Value):
                x = [x]
            out.append(x)

        return out if is_batch else out[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
