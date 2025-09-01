from mAI import MLP, Value, ActivationF, LossF, DataLoader, Optimizer

# =========================
# ===== LossF Testing =====
# =========================
# Suppose model has 3 output logits
logits = [Value(2.0), Value(1.0), Value(0.1)]
target = [0]  # class 0 is correct

loss = LossF.cross_entropy(logits, target)
loss.backward()

print("loss:", loss)
print(logits)

# ===============================
# ===== Training Model Test =====
# ===============================
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

num_epochs = 200
lr = 0.005
batch_size = 3
optim = Optimizer(model.parameters(), lr, 'SGD')
loader = DataLoader(data, targets, batch_size=batch_size, shuffle=True)

for i, (batch_data, batch_target) in enumerate(loader):
    print(f'Batch {i} | X: {batch_data} y: {batch_target}')

for epoch in range(num_epochs):
    avg_loss = 0.0

    for data, target in loader:
        output = model(data)

        loss = LossF.cross_entropy(output, target)

        optim.zero_grad()
        loss.backward()
        optim.update()

        avg_loss += loss.data

    avg_loss /= len(loader)

    print(f'Epoch: [{epoch}/{num_epochs}], Avg Loss: {avg_loss}')
