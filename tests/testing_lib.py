from ..core.value import Value, no_grad
from ..nn.loss import LossF
from ..nn.functional import ActivationF
from ..nn.optim import SGD
from ..data.dataloader import DataLoader
from ..nn.module import MLP

# =========================
# ===== LossF Testing =====
# =========================
def test_lossF():
    # 3-class logits, class 0 correct
    logits = [Value(2.0), Value(1.0), Value(0.1)]
    target = 0
    loss = LossF.cross_entropy(logits, target)
    loss.backward()
    # Ensure gradients flowed to inputs
    assert any(abs(v.grad) > 0 for v in logits)

# ===============================
# ===== Training Model Test =====
# ===============================
def test_training():
    model = MLP(2, [6, 6, 2], ActivationF.relu)
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
    num_epochs = 5
    lr = 0.01
    batch_size = 3
    optim = SGD(model.parameters(), lr)
    loader = DataLoader(data, targets, batch_size=batch_size, shuffle=True)
    first_epoch_loss = None
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
        if first_epoch_loss is None:
            first_epoch_loss = avg_loss
        last_epoch_loss = avg_loss
    # Training should reduce loss
    assert last_epoch_loss < first_epoch_loss

def test_logits_one_hot_inputs_cross_entropy():
    logits = [Value(1.2), Value(3.1), Value(7.3)]
    l_index = LossF.cross_entropy(logits, 1)
    l_onehot = LossF.cross_entropy(logits, [0, 1, 0])
    assert abs(l_index.data - l_onehot.data) < 1e-6

def test_no_grad():
    p = Value(2.1)
    q = Value(-5.1)
    (p * 4).backward()
    with no_grad():
        _ = q * 4
        # backward should be a no-op and no graph linked
        q.backward()  # does nothing
    assert abs(p.grad) > 0
    assert q.grad == 0.0
