from mAI import MLP
from mAI import ActivationF

model = MLP(3, [3], ActivationF.tanh)

print(model([[1.2, 3.2, 5.1]]))
