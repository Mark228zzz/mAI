from ..core.value import Value
from ..nn.loss import LossF
from ..nn.functional import ActivationF
from ..nn.optim import SGD, Adam
from ..core.value import no_grad
from ..data.dataloader import DataLoader
from ..nn.module import MLP
import torch

# =========================
# ===== LossF Testing =====
# =========================
def test_lossF():
    # Suppose model has 3 output logits
    logits = [Value(2.0), Value(1.0), Value(0.1)]
    target = 0  # class 0 is correct

    loss = LossF.cross_entropy(logits, target)
    loss.backward()

    print("loss:", loss)
    print(logits)

# ===============================
# ===== Training Model Test =====
# ===============================
def test_training():
    model = MLP(2, [6, 6, 2], ActivationF.relu)

    print(f'Model\'s parameters amount: {len(model.parameters())}')

    data = [
        [1.2, 3.2],
        [1.5, 3.6],
        [2.1, 4.2],
        [5.2, -1.2],
        [6.2, -2.1],
        [7.8, -0.9],
        [-0.2, 1.5]
    ]

    targets = [0, 0, 0, 1, 1, 1, 0]

    num_epochs = 20
    lr = 0.005
    batch_size = 3
    optim = SGD(model.parameters(), lr)
    loader = DataLoader(data, targets, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        avg_loss = 0.0

        for data, target in loader:
            output = model(data)

            loss = LossF.cross_entropy(output, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss += loss.data

        avg_loss /= len(loader)

        print(f'Epoch: [{epoch}/{num_epochs}], Avg Loss: {avg_loss}')

def test_logits_one_hot_inputs_cross_entropy():
    logits = [Value(1.2), Value(3.1), Value(7.3)]
    y_true = [0, 1, 0]
    y_pred_t = torch.tensor([[1.2, 3.1, 7.3]])
    y_true_t = torch.tensor([1])

    print(LossF.cross_entropy(logits, 1))       # index target -> ~4.2171
    print(LossF.cross_entropy(logits, [0,1,0])) # one-hot target -> ~4.2171

    print(torch.nn.functional.cross_entropy(y_pred_t, y_true_t))
    print(LossF.cross_entropy(logits, y_true))

    print(torch.nn.functional.cross_entropy(torch.tensor([[1.2, 5.1], [4.2, 0.2], [4.2, -2.1]]), torch.tensor([0, 1, 0])))
    print(LossF.cross_entropy([[Value(1.2), Value(5.1)], [Value(4.2), Value(0.2)], [Value(4.2), Value(-2.1)]], [0, 1, 0]))

def test_no_grad():
    param = Value(2.1)
    param_without_grad = Value(-5.1)

    param * 4

    with no_grad():
        param_without_grad * 4

    print(param.grad)
    print(param_without_grad.grad)


test_no_grad()
