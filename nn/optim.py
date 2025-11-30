import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.tensor import Tensor
from typing import List, Literal, Dict
import numpy as np


class Optimizer:
    """
    Base optimizer class for parameter updates during training.

    All optimizers maintain references to model parameters and update them
    in-place during the step() method.
    """

    def __init__(self, parameters: List[Tensor], lr: float = 0.001) -> None:
        """
        Args:
            parameters: List of model parameters (tensors with require_grad=True)
            lr: Learning rate (step size)
        """
        self.parameters = parameters
        self.lr = lr
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Ensure all parameters require gradients and have proper initialization."""
        for i, param in enumerate(self.parameters):
            if not param.require_grad:
                raise ValueError(f"Parameter at index {i} does not require gradients")
            if param.grad is None:
                raise ValueError(f"Parameter at index {i} has not been initialized with gradients")

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero before backward pass."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = np.zeros_like(param.data)

    def step(self) -> None:
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement step()")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set new learning rate."""
        self.lr = lr


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Update rule: theta(t+1) = theta(t) - lr * grad(theta(t))

    Args:
        parameters: List of model parameters
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0.0, no momentum)
        dampening: Dampening for momentum (default: 0.0)
        weight_decay: L2 regularization coefficient (default: 0.0)

    Example:
        >>> model = Sequential(Linear(784, 128), Linear(128, 10))
        >>> optimizer = SGD(model.get_parameters(), lr=0.01, momentum=0.9)
        >>>
        >>> for epoch in range(epochs):
        >>>     optimizer.zero_grad()
        >>>     loss = compute_loss(model(x), y)
        >>>     loss.backward()
        >>>     optimizer.step()
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

        # Initialize momentum buffers if momentum > 0
        self.velocity: Dict[int, np.ndarray] = {}
        if self.momentum > 0:
            for i, param in enumerate(self.parameters):
                self.velocity[i] = np.zeros_like(param.data)

    def step(self) -> None:
        """Perform single SGD step with optional momentum."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Apply momentum if specified
            if self.momentum > 0:
                if i not in self.velocity:
                    self.velocity[i] = np.zeros_like(param.data)

                # v(t) = momentum * v(t-1) + (1 - dampening) * grad(t)
                self.velocity[i] = (
                    self.momentum * self.velocity[i] +
                    (1 - self.dampening) * grad
                )
                grad = self.velocity[i]

            # Update parameters: theta = theta - lr * grad
            param.data -= self.lr * grad


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum and adaptive learning rates per parameter.

    Update rules:
        m(t) = beta1 * m(t-1) + (1 - beta1) * grad(t)              # first moment
        v(t) = beta2 * v(t-1) + (1 - beta2) * grad(t)^2            # second moment
        m_hat(t) = m(t) / (1 - beta1^t)                            # bias correction
        v_hat(t) = v(t) / (1 - beta2^t)                            # bias correction
        theta(t+1) = theta(t) - lr * m_hat(t) / (sqrt(v_hat(t)) + eps)

    Args:
        parameters: List of model parameters
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.0)

    Example:
        >>> optimizer = Adam(model.get_parameters(), lr=0.001)
        >>>
        >>> for epoch in range(epochs):
        >>>     optimizer.zero_grad()
        >>>     loss = compute_loss(model(x), y)
        >>>     loss.backward()
        >>>     optimizer.step()
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m: Dict[int, np.ndarray] = {}  # First moment (mean)
        self.v: Dict[int, np.ndarray] = {}  # Second moment (variance)
        self.t = 0  # Time step

        # Initialize buffers
        for i, param in enumerate(self.parameters):
            self.m[i] = np.zeros_like(param.data)
            self.v[i] = np.zeros_like(param.data)

    def step(self) -> None:
        """Perform single Adam optimization step."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Update biased first moment: m(t) = beta1 * m(t-1) + (1 - beta1) * grad(t)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second moment: v(t) = beta2 * v(t-1) + (1 - beta2) * grad(t)^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment: m_hat(t) = m(t) / (1 - beta1^t)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment: v_hat(t) = v(t) / (1 - beta2^t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters: theta = theta - lr * m_hat(t) / (sqrt(v_hat(t)) + eps)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    """
    RMSprop optimizer - Root Mean Square Propagation.

    Adapts learning rate for each parameter based on recent gradient magnitudes.

    Update rules:
        v(t) = alpha * v(t-1) + (1 - alpha) * grad(t)^2
        theta(t+1) = theta(t) - lr * grad(t) / (sqrt(v(t)) + eps)

    Args:
        parameters: List of model parameters
        lr: Learning rate (default: 0.01)
        alpha: Smoothing constant (default: 0.99)
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.0)
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize squared gradient moving average
        self.v: Dict[int, np.ndarray] = {}
        for i, param in enumerate(self.parameters):
            self.v[i] = np.zeros_like(param.data)

    def step(self) -> None:
        """Perform single RMSprop optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Update squared gradient moving average: v(t) = alpha * v(t-1) + (1 - alpha) * grad(t)^2
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)

            # Update parameters: theta = theta - lr * grad(t) / (sqrt(v(t)) + eps)
            param.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)


def get_optimizer(
    name: Literal['sgd', 'adam', 'rmsprop'],
    parameters: List[Tensor],
    **kwargs
) -> Optimizer:
    """
    Factory function to create optimizers by name.

    Args:
        name: Optimizer name ('sgd', 'adam', 'rmsprop')
        parameters: Model parameters
        **kwargs: Optimizer-specific arguments

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = get_optimizer('adam', model.get_parameters(), lr=0.001)
    """
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop
    }

    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")

    return optimizers[name](parameters, **kwargs)