from __future__ import annotations
from typing import List
from dataclasses import dataclass, field
from scigrad.tensor import UOp

@dataclass(frozen=True)
class KernelSpec:
    uops: List[UOp]
    inputs: dict = field(default_factory=dict)

def schedule(op: UOp) -> List[KernelSpec]:
    """
    Schedules a UOp for execution by creating a single, topologically sorted kernel.
    """
    topo_order = []
    visited = set()

    def _visit(o):
        if o in visited:
            return
        visited.add(o)
        for i in o.inputs:
            if isinstance(i, UOp):
                _visit(i)
        topo_order.append(o)

    _visit(op)

    kernel_uops = [u for u in topo_order if u.op != "LOAD"]
    if not kernel_uops:
        return []

    # All LOAD ops are inputs to this single kernel
    kernel_inputs = {u.uid: u.inputs[0] for u in topo_order if u.op == "LOAD"}

    return [KernelSpec(uops=kernel_uops, inputs=kernel_inputs)]
