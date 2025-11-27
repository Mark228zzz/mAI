from typing import Literal, List
from ..core_old.value import Value


class Optimizer:
    def __init__(
            self,
            parameters: List[Value],
            learning_rate: float,
            method: Literal['SGD'] = 'SGD'
        ):

        self.model_params = list(parameters)
        self.method = method.lower().strip()
        self.a = float(learning_rate)

    def zero_grad(self):
        for p in self.model_params:
            p.grad = 0.0

    def update(self):
        match self.method:
            case 'sgd': self._sgd()
            case _: raise ValueError(f'Optimization method `{self.method}` is not valid')

    def _sgd(self):
        for p in self.model_params:
            p.data += -self.a * p.grad
