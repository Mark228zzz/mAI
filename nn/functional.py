from typing import List
from ..core.value import Value


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
