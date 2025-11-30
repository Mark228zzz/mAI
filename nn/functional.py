import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from core.tensor import Tensor


class LossFunction:
    @staticmethod
    def mse_loss(
        outputs: Tensor,
        targets: Tensor
    ) -> Tensor:
        """
        Compute Mean Squared Error loss between predictions and targets.

        Formula: MSE = (1/n) * sum(outputs - targets)**2

        Args:
            outputs: Predicted values, shape (batch_size, features)
            targets: Ground truth values, shape (batch_size, features)

        Returns:
            Scalar tensor containing the average MSE across all samples

        Example:
            >>> outputs = Tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> targets = Tensor([[1.5, 2.5], [2.8, 3.9]])
            >>> loss = LossFunction.mse_loss(outputs, targets)
        """

        assert outputs.shape == targets.shape, f'Outputs and Targets shapes have to be equal, but got {outputs.shape} and {targets.shape}'
        assert not np.isscalar(outputs.data) and not np.isscalar(targets.data), f'Inputs cannot be scalers but got {outputs} and {targets}'
        assert targets.data.ndim == 2, f'Expected 2D (batch_size, features), got {outputs.shape}'

        diff = outputs - targets
        squared = diff ** 2

        mean_across_samples = squared.mean()

        return mean_across_samples

    @staticmethod
    def cross_entropy_loss(
        outputs: Tensor,
        targets: Tensor
    ):
        """
        Compute Cross-Entropy loss with softmax for multi-class classification.
        Formula: CE = -(1/batch_size) * sum(targets * log(softmax(outputs)))
        Applies softmax to outputs for numerical stability, then computes
        negative log-likelihood of the correct classes.

        Args:
            outputs: Raw logits (pre-softmax), shape (batch_size, num_classes)
            targets: Either:
                    - One-hot encoded labels, shape (batch_size, num_classes)
                    - Class indices, shape (batch_size, 1) or (batch_size,)

        Returns:
            Scalar tensor containing the average cross-entropy loss

        Example:
            >>> # One-hot targets
            >>> outputs = Tensor([[2.0, 1.0], [0.5, 2.5]])  # logits
            >>> targets = Tensor([[1, 0], [0, 1]])          # one-hot
            >>> loss = LossFunction.cross_entropy_loss(outputs, targets)

            >>> # Index targets
            >>> outputs = Tensor([[2.0, 1.0, 0.5], [0.5, 2.5, 1.0]])
            >>> targets = Tensor([[0], [1]])  # or Tensor([0, 1])
            >>> loss = LossFunction.cross_entropy_loss(outputs, targets)
        """
        assert not np.isscalar(outputs.data) and not np.isscalar(targets.data), \
            f'Inputs cannot be scalars but got {outputs} and {targets}'
        assert outputs.data.ndim == 2, \
            f'Outputs must be 2D (batch_size, num_classes), got {outputs.shape}'
        assert targets.shape[0] == outputs.shape[0]

        batch_size, num_classes = outputs.shape

        # Check if targets need one-hot encoding
        one_hotted = targets.shape[-1] == num_classes

        if not one_hotted:
            # Targets are class indices, need to one-hot encode
            # Handle both (batch_size,) and (batch_size, 1) shapes
            target_indices = targets.data.flatten().astype(int)

            # Validate indices
            assert np.all((target_indices >= 0) & (target_indices < num_classes)), \
                f'Target indices must be in range [0, {num_classes}), got min={target_indices.min()}, max={target_indices.max()}'

            # Create one-hot encoded targets
            one_hot = np.zeros((batch_size, num_classes))
            one_hot[np.arange(batch_size), target_indices] = 1
            targets = Tensor(one_hot, require_grad=False)
        else:
            # Targets are already one-hot encoded
            assert targets.shape == outputs.shape, \
                f'One-hot targets shape {targets.shape} must match outputs shape {outputs.shape}'

        # Softmax for numerical stability (log-sum-exp trick)
        exp_outputs = (outputs - outputs.max(axis=1, keepdims=True)).exp()
        probs = exp_outputs / exp_outputs.sum(axis=1, keepdims=True)

        # Cross entropy: -sum(targets * log(probs)) / batch_size
        loss = -(targets * probs.log()).sum() / batch_size

        print(targets)

        return loss

class ActivationFunction:
    @staticmethod
    def relu(x: Tensor) -> Tensor: return x.relu()
    @staticmethod
    def tanh(x: Tensor) -> Tensor: return x.tanh()
    @staticmethod
    def sigmoid(x: Tensor) -> Tensor: return x.sigmoid()
    @staticmethod
    def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor: return x.leaky_relu(alpha=alpha)
