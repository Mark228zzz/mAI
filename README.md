# mAI - Educational Deep Learning Library

A lightweight, NumPy-based deep learning library built from scratch for educational purposes. Think of it as a simplified PyTorch/TensorFlow that helps you understand what's happening under the hood.

## Inspiration & Acknowledgments

This project was heavily inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), which is an excellent resource for understanding automatic differentiation and neural networks at a fundamental level. His work has been instrumental in helping me (and countless others) grasp the core concepts of deep learning frameworks.

**Big thanks to Andrej Karpathy** for creating such an accessible and elegant teaching tool that demystifies backpropagation and autograd engines!

## Important Note

⚠️ **This library is NOT production-ready and is NOT optimized for performance.** It's purely educational and designed to help understand:
- How autograd engines work
- How neural networks are built from scratch
- How optimization algorithms update parameters
- The mathematics behind backpropagation

For real projects, use PyTorch, TensorFlow, or JAX.

## Features

- **Automatic Differentiation**: Full computational graph with automatic gradient computation
- **Neural Network Layers**: Linear layers with various initialization methods (He, Xavier)
- **Activation Functions**: ReLU, Tanh, Sigmoid, Leaky ReLU
- **Loss Functions**: MSE, Cross-Entropy (with automatic one-hot encoding)
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **NumPy Backend**: Simple and readable implementation

## Project Structure

```
mAI/
├── core/
│   ├── tensor.py          # Tensor class with autograd
│   └── module.py          # Base Module class
├── nn/
│   ├── nn.py             # Linear, Sequential layers
│   ├── functional.py     # Activation & Loss functions
│   └── optim.py          # Optimizers (SGD, Adam, RMSprop)
└── tests/
    ├── test_tensor.py    # Unit tests for Tensor
    └── test_training.py  # Integration tests
```

## Quick Start

```python
from nn.nn import Sequential, Linear
from nn.functional import ActivationFunction as af, LossFunction as lf
from nn.optim import get_optimizer
from core.tensor import Tensor

# Create a simple neural network
model = Sequential(
    Linear(784, 128, af.relu),
    Linear(128, 64, af.relu),
    Linear(64, 10)
)

# Prepare data
x_train = Tensor.randn((32, 784))  # 32 samples, 784 features
y_train = Tensor.randn((32, 10))   # 32 samples, 10 classes

# Setup optimizer
optimizer = get_optimizer('adam', model.get_parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(x_train)
    loss = lf.mse_loss(output, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.data:.4f}')
```

## Examples

### Simple Linear Regression

```python
# Learn y = 2x + 1
data = Tensor([[1.0], [2.0], [3.0], [4.0]])
target = Tensor([[3.0], [5.0], [7.0], [9.0]])

model = Sequential(
    Linear(1, 32, af.relu),
    Linear(32, 1)
)

optimizer = get_optimizer('sgd', model.get_parameters(), lr=0.01, momentum=0.9)

for epoch in range(500):
    output = model(data)
    loss = lf.mse_loss(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Classification with Cross-Entropy

```python
# Multi-class classification
model = Sequential(
    Linear(10, 64, af.relu),
    Linear(64, 32, af.relu),
    Linear(32, 3)  # 3 classes
)

# Targets can be indices or one-hot encoded
targets = Tensor([[0], [1], [2], [0]])  # Class indices
# OR
targets = Tensor([[1,0,0], [0,1,0], [0,0,1], [1,0,0]])  # One-hot

logits = model(x_train)
loss = lf.cross_entropy_loss(logits, targets)
```

## Available Optimizers

```python
# SGD with momentum
optimizer = get_optimizer('sgd', params, lr=0.01, momentum=0.9)

# Adam (default choice)
optimizer = get_optimizer('adam', params, lr=0.001, betas=(0.9, 0.999))

# RMSprop
optimizer = get_optimizer('rmsprop', params, lr=0.01, alpha=0.99)
```

## Tensor Operations

```python
# Create tensors
a = Tensor([1.0, 2.0, 3.0], require_grad=True)
b = Tensor.randn((3, 3), require_grad=True)
c = Tensor.zeros((5, 5))

# Math operations
y = a + b
y = a * b
y = a @ b  # Matrix multiplication
y = a ** 2
y = a.exp()
y = a.log()

# Activations
y = a.relu()
y = a.tanh()
y = a.sigmoid()

# Reductions
y = a.sum()
y = a.mean()
y = a.max()

# Backpropagation
y.backward()
print(a.grad)  # Gradients of a
```

## Running Tests

```bash
# Test tensor operations
python tests/test_tensor.py

# Test full training loop
python tests/test_training.py
```

## What I Learned

Building this library taught me:
- How automatic differentiation works (chain rule in practice)
- The relationship between forward and backward passes
- How broadcasting works in gradient computation
- Why bias correction is needed in Adam
- The importance of proper weight initialization
- How modern deep learning frameworks are structured

## License

MIT License - Feel free to use this for learning!

## Contributing

This is primarily an educational project, but if you find bugs or want to add features for learning purposes, feel free to open an issue or PR.

---

**Remember**: This is a learning tool, not a production library. Use PyTorch, TensorFlow, or JAX for real projects! 🚀
