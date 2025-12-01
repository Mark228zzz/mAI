import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.tensor import Tensor
import numpy as np


def test_addition():
    '''Test tensor addition and gradient backprop.'''
    a = Tensor([2.0], require_grad=True)
    b = Tensor([3.0], require_grad=True)
    c = a + b
    c.backward()

    assert c.data[0] == 5.0
    assert a.grad[0] == 1.0
    assert b.grad[0] == 1.0
    print('Addition test passed')


def test_multiplication():
    '''Test tensor multiplication.'''
    a = Tensor([2.0], require_grad=True)
    b = Tensor([3.0], require_grad=True)
    c = a * b
    c.backward()

    assert c.data[0] == 6.0
    assert a.grad[0] == 3.0  # dc/da = b
    assert b.grad[0] == 2.0  # dc/db = a
    print('Multiplication test passed')


def test_exp_backward():
    '''Test exponential backward pass.'''
    x = Tensor([1.0], require_grad=True)
    y = x.exp()
    y.backward()

    # d(e^x)/dx = e^x
    assert np.isclose(y.data[0], np.e)
    assert np.isclose(x.grad[0], np.e)
    print('Exp backward test passed')


if __name__ == '__main__':
    test_addition()
    test_multiplication()
    test_exp_backward()
    print('\nAll tensor tests passed!')
