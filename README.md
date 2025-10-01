# mAI — tiny autograd and neural nets in pure Python

mAI is a minimal educational library for building and training small neural networks from scratch, inspired by Andrej Karpathy’s micrograd and his video “The spelled‑out intro to neural networks and backpropagation”.

- Video (**highly recommended**): https://www.youtube.com/watch?v=VMj-3S1tku0
- micrograd repo: https://github.com/karpathy/micrograd

The goal is clarity over speed: a tiny `Value` class drives reverse‑mode autodiff, simple `Module`s compose neurons/layers/MLPs, and a couple of optimizers make training straightforward.

## Features
- Scalar autograd via `Value` (reverse‑mode backprop)
- Lightweight `Module` hierarchy (`Neuron`, `Layer`, `MLP`)
- Loss functions: MSE, Cross‑Entropy (with softmax)
- Optimizers: `SGD`, `Adam`
- `Parameter` type to mark learnable tensors (subclass of `Value`)
- Simple `DataLoader` for batching
- Optional `no_grad()` context for inference (skip graph building)

## Install
This repository is designed to be read and run directly. A Python 3.10+ environment is recommended.

## Quickstart
Below is a tiny classification example with the provided MLP and optimizers.

```python
from mAI import MLP, SGD, Adam, ActivationF, LossF, DataLoader

# Toy data (2D → 2 classes)
data = [
    [1.2, 3.2], [1.5, 3.6], [2.1, 4.2],
    [5.2, -1.2], [6.2, -2.1], [7.8, -0.9],
]
targets = [0, 0, 0, 1, 1, 1]

model = MLP(2, [6, 6, 2], ActivationF.relu)
loader = DataLoader(data, targets, batch_size=3, shuffle=True)

# Choose an optimizer
opt = SGD(model.parameters(), lr=5e-3)  # or Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    avg_loss = 0.0
    for xb, yb in loader:
        logits = model(xb)
        loss = LossF.cross_entropy(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_loss += loss.data
    avg_loss /= len(loader)
    print(f"epoch {epoch:02d}  loss {avg_loss:.4f}")
```

## Key Concepts
- `Value`: scalar holding `data` and `grad`, plus overloaded ops for graph building and backprop.
- `Parameter`: just a tagged `Value` used for model weights/biases.
- `Module`: base class for models with `parameters()` and `zero_grad()`; subclasses include `Neuron`, `Layer`, `MLP`.
- `LossF`: contains `mse` and `cross_entropy` (uses softmax internally).
- `SGD` / `Adam`: step and zero_grad interfaces similar to PyTorch, with optional `weight_decay`.
- `no_grad()`: context manager to disable graph construction during inference.

## File Structure
- `core/value.py` — `Value`, and `no_grad()`
- `core/parameter.py` — `Parameter`
- `core/module.py` — `Module` base
- `nn/module.py` — `Neuron`, `Layer`, `MLP`
- `nn/functional.py` — activations and softmax
- `nn/loss.py` — MSE and Cross‑Entropy losses
- `nn/optim.py` — `SGD`, `Adam` (and a simple legacy `Optimizer` wrapper)
- `data/dataloader.py` — tiny `DataLoader`

## Notes
- This project mirrors the educational spirit of micrograd; it’s intentionally simple and not optimized.
- The API aims to be familiar to PyTorch users while remaining tiny and easy to read.

## Acknowledgments
Huge thanks to Andrej Karpathy for micrograd and the fantastic backpropagation walkthrough that inspired this project.
