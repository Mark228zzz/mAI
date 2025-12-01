import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data import split_tensors
from core.tensor import Tensor
import numpy as np


def test_split_basic():
    '''Test basic split functionality.'''
    data = Tensor.randn((100, 10))
    target = Tensor.randn((100, 5))

    result = split_tensors(data, target, test_ratio=0.2)

    assert len(result) == 4
    data_train, target_train, data_test, target_test = result

    assert data_train.shape == (80, 10)
    assert data_test.shape == (20, 10)
    assert target_train.shape == (80, 5)
    assert target_test.shape == (20, 5)

    print('Basic split test passed')


def test_split_ratios():
    '''Test different split ratios.'''
    data = Tensor.randn((1000, 50))
    target = Tensor.randn((1000, 10))

    result = split_tensors(data, target, test_ratio=0.3)
    data_train, _, data_test, _ = result

    assert data_train.shape[0] == 700
    assert data_test.shape[0] == 300

    print('Split ratio test passed')


def test_no_overlap():
    '''Test that train and test sets don't overlap.'''
    data = Tensor(np.arange(100).reshape(100, 1).astype(float))
    target = Tensor(np.arange(100).reshape(100, 1).astype(float) * 2)

    result = split_tensors(data, target, test_ratio=0.2)
    data_train, target_train, data_test, target_test = result

    train_values = set(data_train.data.flatten())
    test_values = set(data_test.data.flatten())
    overlap = train_values.intersection(test_values)

    assert len(overlap) == 0

    # Check correspondence is maintained
    for i in range(len(data_train.data)):
        assert np.isclose(target_train.data[i], data_train.data[i] * 2)

    print('No overlap test passed')


if __name__ == '__main__':
    test_split_basic()
    test_split_ratios()
    test_no_overlap()
    print('\nAll tests passed!')
