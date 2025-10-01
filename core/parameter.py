from .value import Value
from typing import Optional, Tuple


class Parameter(Value):
    """A tagged Value that represents a learnable parameter."""
    def __init__(self, data: float | int, _children: Optional[Tuple['Value', ...]] | None = (), _op: str = '', label: str = 'Parameter'):
        super().__init__(data=data, _children=_children, _op=_op, label=label)
