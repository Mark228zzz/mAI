import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data import DataLoader
from core.tensor import Tensor


def test_two_tensors():
    '''Test with two tensors (standard case).'''
    data = Tensor.randn((100, 10))
    target = Tensor.randn((100, 5))

    loader = DataLoader(data, target, batch_size=32)

    assert len(loader) == 4

    for batch_data, batch_target in loader:
        assert batch_data.shape[1] == 10
        assert batch_target.shape[1] == 5

    print('Two tensors test passed')


def test_three_tensors():
    '''Test with three tensors.'''
    data = Tensor.randn((100, 10))
    target = Tensor.randn((100, 5))
    weights = Tensor.randn((100, 1))

    loader = DataLoader(data, target, weights, batch_size=32)

    for batch_data, batch_target, batch_weights in loader:
        assert batch_data.shape[1] == 10
        assert batch_target.shape[1] == 5
        assert batch_weights.shape[1] == 1

    print('Three tensors test passed')


def test_single_tensor():
    '''Test with single tensor.'''
    data = Tensor.randn((100, 10))

    loader = DataLoader(data, batch_size=32)

    for (batch_data,) in loader:
        assert batch_data.shape[1] == 10

    print('Single tensor test passed')


def test_mismatched_shapes():
    '''Test that mismatched shapes raise error.'''
    data = Tensor.randn((100, 10))
    target = Tensor.randn((50, 5))  # Wrong number of samples

    try:
        loader = DataLoader(data, target, batch_size=32)
        assert False, 'Should have raised assertion error'
    except AssertionError as e:
        assert 'same number of samples' in str(e)

    print('Mismatched shapes test passed')

if __name__ == '__main__':
    test_two_tensors()
    test_three_tensors()
    test_single_tensor()
    test_mismatched_shapes()
    print('\nAll DataLoader tests passed!')
