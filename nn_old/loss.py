from typing import List
from ..core_old.value import Value
from ..nn_old.functional import Functional


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
