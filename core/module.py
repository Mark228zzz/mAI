from abc import ABC, abstractmethod


class Module(ABC):
    """
    Base class for all  layers.
    Subclasses must implement `forward`.
    """

    def get_parameters(self):
        return []

    def zero_grad(self):
        for p in self.get_parameters():
            p.grad = 0.0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def init(self):
        ...

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...
