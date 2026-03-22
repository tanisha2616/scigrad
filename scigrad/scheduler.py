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
    Schedules a UOp into topologically sorted kernel groups.

    Fusion policy (current):
    - Adjacent non-boundary ops are grouped together.
    - Reduction/materialization ops terminate a group.
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

    non_load_uops = [u for u in topo_order if u.op != "LOAD"]
    if not non_load_uops:
        return []

    kernel_inputs = {u.uid: u.inputs[0] for u in topo_order if u.op == "LOAD"}

    boundary_ops = {"REDUCE_SUM", "REDUCE_MAX", "REDUCE_PROD", "CONTIGUOUS"}

    kernels: List[KernelSpec] = []
    current_group: List[UOp] = []

    for u in non_load_uops:
        current_group.append(u)
        if u.op in boundary_ops:
            kernels.append(KernelSpec(uops=current_group.copy(), inputs=kernel_inputs))
            current_group = []

    if current_group:
        kernels.append(KernelSpec(uops=current_group, inputs=kernel_inputs))

    return kernels
