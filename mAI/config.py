from .nn import Value
import math


class Functional:
    @staticmethod
    def softmax(data):
        if not isinstance(data, list):
            data = [x.data for x in data]

        e_sum = sum(math.exp(x) for x in data)
        result = [math.exp(x) / e_sum for x in data]

        return result

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
        # Convert float and int into Value object
        x = x if isinstance(x, Value) else Value(x)

        t = ((x*2).exp() - 1) / ((x*2).exp() + 1)

        out = Value(t, (x, ), 'tanh')

        # Define backpropagation rule + apply chain rule (out.grad)
        def _backward():
            x.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    @staticmethod
    def relu(x):
        out = Value(x.data if x.data > 0 else 0, (x,), 'relu')

        def _backward():
            x.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out


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
