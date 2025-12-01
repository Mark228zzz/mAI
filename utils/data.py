import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.tensor import Tensor
import numpy as np
from typing import Tuple, Generator, List


class DataLoader:
    """
    DataLoader for batching and iterating over dataset.

    Args:
        *tensors: Variable number of input tensors (data, target, weights, etc.)
                  All tensors must have the same number of samples (first dimension)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data each epoch
        drop_last: Whether to drop the last incomplete batch

    Example:
        >>> # Two tensors (data and target)
        >>> data = Tensor.randn((100, 784))
        >>> target = Tensor.randn((100, 10))
        >>> loader = DataLoader(data, target, batch_size=32, shuffle=True)
        >>>
        >>> for batch_data, batch_target in loader:
        >>>     output = model(batch_data)
        >>>     loss = loss_fn(output, batch_target)

        >>> # Three tensors (data, target, weights)
        >>> data = Tensor.randn((100, 784))
        >>> target = Tensor.randn((100, 10))
        >>> weights = Tensor.randn((100, 1))
        >>> loader = DataLoader(data, target, weights, batch_size=32)
        >>>
        >>> for batch_data, batch_target, batch_weights in loader:
        >>>     output = model(batch_data)
        >>>     loss = loss_fn(output, batch_target) * batch_weights
    """

    def __init__(
        self,
        *tensors: Tensor,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        assert len(tensors) > 0, "At least one tensor must be provided"
        assert batch_size > 0, "Batch size must be positive"

        # Check all tensors have the same number of samples
        num_samples = tensors[0].shape[0]
        for i, tensor in enumerate(tensors):
            assert tensor.shape[0] == num_samples, \
                f"All tensors must have same number of samples. " \
                f"Tensor 0 has {num_samples} samples, but tensor {i} has {tensor.shape[0]}"

        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = num_samples
        self.num_batches = self._calculate_num_batches()

    def _calculate_num_batches(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Generator[Tuple[Tensor, ...], None, None]:
        # Generate indices
        indices = np.arange(self.num_samples)

        # Shuffle if required
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            # Skip incomplete batch if drop_last is True
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                break

            batch_indices = indices[start_idx:end_idx]

            # Extract batch for each tensor
            batch_tensors = []
            for tensor in self.tensors:
                batch_tensor = Tensor(
                    tensor.data[batch_indices],
                    require_grad=False
                )
                batch_tensors.append(batch_tensor)

            yield tuple(batch_tensors)

    def get_batch(self, batch_idx: int) -> Tuple[Tensor, ...]:
        """
        Get a specific batch by index.

        Args:
            batch_idx: Index of the batch to retrieve

        Returns:
            Tuple of batch tensors
        """
        if batch_idx < 0 or batch_idx >= self.num_batches:
            raise IndexError(f"Batch index {batch_idx} out of range [0, {self.num_batches})")

        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        batch_tensors = []
        for tensor in self.tensors:
            batch_tensor = Tensor(
                tensor.data[start_idx:end_idx],
                require_grad=False
            )
            batch_tensors.append(batch_tensor)

        return tuple(batch_tensors)


def split_tensors(*tensors: Tensor, test_ratio: float = 0.2) -> Tuple[Tuple[Tensor], Tuple[Tensor]]:
    """
    Split tensors into train and test tensors.

    Args:
        train_ratio: Ratio of training data (default: 0.8)

    Returns:
        Tuple of ([train tensors], [test tensors])

    Example:
        >>> data = Tensor.randn((100, 784))
        >>> target = Tensor.randn((100, 10))
        >>> data_train, target_train, data_test, target_test = split_tensors(data, target, test_ratio=0.2)
    """
    assert 0 < test_ratio < 1, "test_ratio must be between 0 and 1"

    num_samples = tensors[0].shape[0]

    split_idx = int(num_samples * test_ratio)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[split_idx:]
    test_indices = indices[:split_idx]

    # Create train tensors split
    train = [Tensor(tensor.data[train_indices], require_grad=False) for tensor in tensors]

    # Create test tensors split
    test = [Tensor(tensor.data[test_indices], require_grad=False) for tensor in tensors]

    train.extend(test)

    return train
