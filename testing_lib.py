from mAI import MLP, Value, ActivationF, LossF

# =========================
# ===== LossF Testing =====
# =========================
# Suppose model has 3 output logits
logits = [Value(2.0), Value(1.0), Value(0.1)]
target = [0]  # class 0 is correct

loss = LossF.cross_entropy(logits, target)
loss.backward()

print("loss:", loss)
for p in logits:
    print(p, p.grad)
