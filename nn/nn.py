import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.module import Module
from core.tensor import Tensor
from .functional import LossFunction as lf
from .functional import ActivationFunction as af
from typing import Callable, List, Literal


class Linear(Module):
    def __init__(
        self,
        in_feautres: int,
        out_features: int,
        act_func: Callable | None = None,
        init_method: Literal['randn', 'he', 'xavier'] = 'randn'
    ) -> None:
        self.in_f = in_feautres
        self.out_f = out_features
        self.act_func = act_func
        self.init_method = init_method

        # Initiate parameters
        self.init()

    def init(self) -> None:
        if self.init_method == 'he':
            self.weight = Tensor.he_normal((self.out_f, self.in_f), self.in_f, require_grad=True)
            self.bias = Tensor.zeros((self.out_f,), require_grad=True)
        elif self.init_method == 'xavier':
            self.weight = Tensor.xavier_uniform((self.out_f, self.in_f), self.in_f, self.out_f, require_grad=True)
            self.bias = Tensor.zeros((self.out_f,), require_grad=True)
        else:
            self.weight = Tensor.randn((self.out_f, self.in_f), require_grad=True)
            self.bias = Tensor.zeros((self.out_f,), require_grad=True)

    def get_parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        assert x.data.ndim == 2

        out = (x @ self.weight.T()) + self.bias

        if self.act_func: out = self.act_func(out)

        return out


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers

    def init(self):
        for layer in self.layers:
            layer.init()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x

    def get_parameters(self) -> List[Tensor]:
        params = []

        for layer in self.layers:
            params.extend(layer.get_parameters())

        return params
