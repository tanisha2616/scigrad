from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import itertools

# Forward declaration to avoid circular import
class UOp:
    pass

_kid_counter = itertools.count()

@dataclass() # Not frozen anymore to allow mutation during fusion
class KernelSpec:
    """Describes a single kernel to be executed by a backend."""
    op_list: List[UOp]
    input_nodes: List[UOp]
    output_node: UOp
    kid: int = field(default_factory=lambda: next(_kid_counter), init=False)

    def __hash__(self):
        return self.kid

    def __eq__(self, other):
        return self.kid == other.kid
