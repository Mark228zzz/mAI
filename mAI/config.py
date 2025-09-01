from .nn import Value
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

        loss = sum((p - y)**2 for y, p in zip(targets, outputs))

        return loss

    @staticmethod
    def cross_entropy(outputs, targets):
        assert len(outputs) == len(targets), 'Targets length are not equal to outputs length'

        outputs = Functional.softmax(outputs)
        targets = Functional.one_hot(targets)

        loss = -(sum(y * math.log(p.data) for y, p in zip(targets, outputs)) / len(targets))

        return loss
