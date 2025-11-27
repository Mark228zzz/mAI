from abc import ABC, abstractmethod


class Module(ABC):
    """
    Base class for all neurons / layers / models.
    Subclasses must implement `forward`.
    """

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...
