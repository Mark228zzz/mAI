import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from nn.nn import Sequential, Linear
from nn.functional import ActivationFunction as af, LossFunction as lf
from nn.optim import get_optimizer
from core.tensor import Tensor


def test_full_training():
    """Integration test: Train a simple regression model."""
    model = Sequential(
        Linear(1, 1024, af.relu),
        Linear(1024, 512, af.relu),
        Linear(512, 1)
    )

    data = Tensor.randn((4, 1))
    target = Tensor.randn((4, 1))

    epochs = 250
    lr = 0.01

    optimizer = get_optimizer('sgd', model.get_parameters(), lr=lr)

    for epoch in range(epochs):
        output = model(data)
        loss = lf.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:  # Print every 50 epochs
            print(f'Epoch: [{epoch+1}/{epochs}], Loss: {loss.data:.6f}')

    # Assert that loss decreased significantly
    final_output = model(data)
    final_loss = lf.mse_loss(final_output, target)
    assert final_loss.data < 1.0, "Training failed: loss did not decrease"
    print(f'Training test passed! Final loss: {final_loss.data:.6f}')


def test_optimizer_comparison():
    """Compare different optimizers on the same task."""
    optimizers_to_test = ['sgd', 'adam', 'rmsprop']

    for opt_name in optimizers_to_test:
        print(f"\nTesting {opt_name.upper()} optimizer:")

        model = Sequential(
            Linear(1, 64, af.relu),
            Linear(64, 1)
        )

        data = Tensor([[1.0], [2.0], [3.0], [4.0]])
        target = Tensor([[2.0], [4.0], [6.0], [8.0]])  # y = 2x

        optimizer = get_optimizer(opt_name, model.get_parameters(), lr=0.01)

        for epoch in range(100):
            output = model(data)
            loss = lf.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'{opt_name.upper()} final loss: {loss.data:.6f}')


if __name__ == '__main__':
    test_full_training()
    test_optimizer_comparison()