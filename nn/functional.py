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
            targets: One-hot encoded labels, shape (batch_size, num_classes)

        Returns:
            Scalar tensor containing the average cross-entropy loss

        Example:
            >>> outputs = Tensor([[2.0, 1.0], [0.5, 2.5]])  # logits
            >>> targets = Tensor([[1, 0], [0, 1]])          # one-hot
            >>> loss = LossFunction.cross_entropy_loss(outputs, targets)
        """
        assert outputs.shape == targets.shape, f'Outputs and Targets shapes have to be equal, but got {outputs.shape} and {targets.shape}'
        assert not np.isscalar(outputs.data) and not np.isscalar(targets.data), f'Inputs cannot be scalers but got {outputs} and {targets}'
        assert targets.data.ndim == 2, f'Expected 2D (batch_size, features), got {outputs.shape}'

        batch_size = outputs.shape[0]

        # Softmax for numerical stability
        exp_outputs = (outputs - outputs.max(axis=1, keepdims=True)).exp()
        probs = exp_outputs / exp_outputs.sum(axis=1, keepdims=True)

        # Cross entropy
        loss = -(targets * probs.log()).sum() / batch_size
        return loss
