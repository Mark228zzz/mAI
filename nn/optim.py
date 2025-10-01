from typing import Iterable, Dict
from ..core.value import Value

class SGD:
    def __init__(self, params: Iterable[Value], lr: float = 1e-2, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        for p in self.params:
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            p.data += -self.lr * g


class Adam:
    def __init__(
        self,
        params: Iterable[Value],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        # Per-parameter state
        self.m: Dict[Value, float] = {p: 0.0 for p in self.params}
        self.v: Dict[Value, float] = {p: 0.0 for p in self.params}
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr = self.lr
        for p in self.params:
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            m = self.m[p] = b1 * self.m[p] + (1 - b1) * g
            v = self.v[p] = b2 * self.v[p] + (1 - b2) * (g * g)
            m_hat = m / (1 - (b1 ** self.t))
            v_hat = v / (1 - (b2 ** self.t))
            p.data += -lr * (m_hat / ((v_hat ** 0.5) + self.eps))
