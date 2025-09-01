from .nn import Value
import random
import math


class Functional:
    @staticmethod
    def softmax(values: list[Value]):
        exps = [v.exp() for v in values]
        e_sum = sum(exps, Value(0.0))
        return [e / e_sum for e in exps]

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
    def tanh(x: Value | float):
        x = x if isinstance(x, Value) else Value(x)

        return x.tanh()

    @staticmethod
    def relu(x: Value | float):
        x = x if isinstance(x, Value) else Value(x)

        return x.relu()


class LossF:
    @staticmethod
    def mse(outputs, targets):
        assert len(outputs) == len(targets), 'Targets length are not equal to outputs length'

        n = len(outputs)

        loss = sum((p - y)**2 for y, p in zip(targets, outputs)) / n

        return loss

    @staticmethod
    def cross_entropy(outputs, targets):
        # Case 1: single sample
        if isinstance(outputs[0], Value):
            probs = Functional.softmax(outputs)
            losses = [-probs[t].log() for t in targets]
            return sum(losses, Value(0.0)) / len(losses)

        # Case 2: batch of samples
        batch_losses = [LossF.cross_entropy(o, [t]) for o, t in zip(outputs, targets)]
        return sum(batch_losses, Value(0.0)) / len(batch_losses)


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
