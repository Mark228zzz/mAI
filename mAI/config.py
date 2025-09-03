from .nn import Value
import random
from typing import List
import math


class Functional:
    @staticmethod
    def softmax(values: List[Value]) -> List[Value]:
        m = max(v.data for v in values)
        shifted = [v - m for v in values]
        exps = [v.exp() for v in shifted]

        total = exps[0]
        for e in exps[1:]:
            total = total + e

        return [e / total for e in exps]

    @staticmethod
    def one_hot(data):
        encodings = {}

        n_unique = len(set(data))

        for i, x in enumerate(set(data)):
            encoding = [1 if i == iu else 0 for iu in range(n_unique)]
            encodings[x] = encoding

        one_hot_list = []

        for x in data:
            one_hot_list.append(encodings[x])

        return one_hot_list


class ActivationF:
    @staticmethod
    def tanh(x: Value | float) -> Value:
        x = x if isinstance(x, Value) else Value(x)

        return x.tanh()

    @staticmethod
    def relu(x: Value | float) -> Value:
        x = x if isinstance(x, Value) else Value(x)

        return x.relu()


class LossF:
    @staticmethod
    def mse(
        outputs: List[Value] | List[List[Value]],
        targets: List[Value] | List[List[Value]]
        ) -> Value | List[Value]:

        assert len(outputs) == len(targets), 'Outputs length has to match length of targets'

        # Case 1: single sample (List[Value])
        if isinstance(outputs[0], Value):
            loss = sum((p - y)**2 for y, p in zip(targets, outputs)) / len(targets)
            return loss

        # Case 2: batch of samples (List[List[Value]])
        batch_losses = [LossF.mse(o, [t]) for o, t in zip(outputs, targets)]
        return sum(batch_losses, Value(0.0)) / len(batch_losses)

    @staticmethod
    def cross_entropy(
        logits: List[Value] | List[List[Value]],
        targets: List[Value] | List[List[Value]]
        ) -> Value | List[Value]:
        """
        logits:
          - single: List[Value] of length C
          - batch : List[List[Value]], each length C
        targets:
          - single: int (class index) OR list/tuple of floats (one-hot / soft)
          - batch : List[int] or List[List[float]]
        returns: mean loss over batch (scalar Value)
        """

        # ----- batch case -----
        if isinstance(logits[0], list):
            assert isinstance(targets, (list, tuple)), "batch targets must be list/tuple"
            assert len(logits) == len(targets), "batch size mismatch"
            loss = LossF.cross_entropy(logits[0], targets[0])
            for x, y in zip(logits[1:], targets[1:]):
                loss = loss + LossF.cross_entropy(x, y)
            return loss / len(logits)

        #  ----- Single sample case -----
        probs = Functional.softmax(logits)

        if isinstance(targets, (int, Value)):
            # single class index
            return -(probs[targets].log())

        if isinstance(targets, list) and len(targets) == len(logits):
            # one-hot or soft distribution
            loss = (-probs[0].log()) * Value(float(targets[0]))
            for p, t in zip(probs[1:], targets[1:]):
                loss = loss + (-p.log()) * Value(float(t))
            return loss

        raise TypeError("target must be int (class index) or list of length C")


class DataLoader:
    def __init__(self, data, target, batch_size=1, shuffle=True):
        assert len(data) == len(target), "data and target must be the same length"

        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration

        # Get batch indices
        batch_idx = self.indices[self.current:self.current+self.batch_size]
        self.current += self.batch_size

        # Slice data + targets
        batch_data = [self.data[i] for i in batch_idx]
        batch_target = [self.target[i] for i in batch_idx]

        return batch_data, batch_target

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
