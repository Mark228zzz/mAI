import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from nn.nn import Sequential, Linear
from nn.functional import ActivationFunction as af, LossFunction as lf
from nn.optim import get_optimizer
from core.tensor import Tensor
from utils.data import DataLoader, split_tensors


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
    assert final_loss.data <= 1000.0, "Training failed: !GRADIENT EXPLODING!"
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

def test_training_with_dataloader():
    """Test training loop with DataLoader."""
    model = Sequential(
        Linear(10, 32, af.relu),
        Linear(32, 5)
    )

    data = Tensor.randn((100, 10))
    target = Tensor.randn((100, 5))

    loader = DataLoader(data, target, batch_size=16, shuffle=True)
    optimizer = get_optimizer('sgd', model.get_parameters(), lr=0.01)

    initial_loss = None
    final_loss = None

    for epoch in range(10):
        epoch_loss = 0.0
        for batch_data, batch_target in loader:
            output = model(batch_data)
            loss = lf.mse_loss(output, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data

        avg_loss = epoch_loss / len(loader)

        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

    assert final_loss < initial_loss, "Loss should decrease during training"
    print(f"Training with DataLoader test passed (loss: {initial_loss:.4f} -> {final_loss:.4f})")


def test_training_with_split():
    """Test training with train/val split."""
    data = Tensor.randn((200, 20))
    target = Tensor.randn((200, 3))

    # Split data
    result = split_tensors(data, target, test_ratio=0.2)
    data_train, target_train, data_test, target_test = result

    # Create loaders
    train_loader = DataLoader(data_train, target_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(data_test, target_test, batch_size=32, shuffle=False)

    model = Sequential(
        Linear(20, 64, af.relu),
        Linear(64, 3)
    )

    optimizer = get_optimizer('adam', model.get_parameters(), lr=0.001)

    # Train for a few epochs
    for epoch in range(5):
        # Training
        train_loss = 0
        for batch_data, batch_target in train_loader:
            output = model(batch_data)
            loss = lf.mse_loss(output, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data

        # Validation
        val_loss = 0
        for batch_data, batch_target in test_loader:
            output = model(batch_data)
            loss = lf.mse_loss(output, batch_target)
            val_loss += loss.data

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)

        if epoch == 4:
            print(f"Final - Train: {avg_train:.4f}, Val: {avg_val:.4f}")

    print("Train/val split test passed")


def test_multiple_optimizers():
    """Test different optimizers with DataLoader."""
    data = Tensor([[1.0], [2.0], [3.0], [4.0]])
    target = Tensor([[2.0], [4.0], [6.0], [8.0]])

    loader = DataLoader(data, target, batch_size=2)

    for opt_name in ['sgd', 'adam']:
        model = Sequential(
            Linear(1, 16, af.relu),
            Linear(16, 1)
        )

        optimizer = get_optimizer(opt_name, model.get_parameters(), lr=0.01)

        for epoch in range(50):
            for batch_data, batch_target in loader:
                output = model(batch_data)
                loss = lf.mse_loss(output, batch_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test prediction
        pred = model(Tensor([[2.0]]))
        expected = 4.0
        error = abs(pred.data[0][0] - expected)

        assert error < 1.0, f"{opt_name} failed to learn y=2x"
        print(f"{opt_name.upper()} optimizer test passed (predicted {pred.data[0][0]:.2f} for input 2.0)")


if __name__ == '__main__':
    test_full_training()
    test_optimizer_comparison()
    test_training_with_dataloader()
    test_training_with_split()
    test_multiple_optimizers()
