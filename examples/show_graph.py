from __future__ import annotations

from collections import deque
import numpy as np
import scigrad.numpy as sgnp
from scigrad.tensor import Tensor, UOp


def _collect_uops(root: UOp):
    visited = set()
    order = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if not isinstance(node, UOp) or node.uid in visited:
            continue
        visited.add(node.uid)
        order.append(node)
        for inp in node.inputs:
            if isinstance(inp, UOp):
                queue.append(inp)

    return order


def to_mermaid(tensor: Tensor) -> str:
    root = tensor._op
    nodes = _collect_uops(root)

    lines = ["graph TD"]
    for node in nodes:
        label = f"{node.uid}:{node.op}\\nshape={node.shape}"
        lines.append(f"  n{node.uid}[\"{label}\"]")

    for node in nodes:
        for inp in node.inputs:
            if isinstance(inp, UOp):
                lines.append(f"  n{inp.uid} --> n{node.uid}")

    return "\n".join(lines)


def print_graph_summary(tensor: Tensor) -> None:
    root = tensor._op
    nodes = _collect_uops(root)
    print("=== Graph summary ===")
    print("Root:", f"uid={root.uid}, op={root.op}, shape={root.shape}")
    print("Node count:", len(nodes))
    print("Nodes:")
    for n in nodes:
        print(f"  uid={n.uid:>4}  op={n.op:<10} shape={n.shape}")


def main() -> None:
    a = sgnp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = sgnp.array(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    expr = sgnp.sum(sgnp.multiply(sgnp.add(a, b), b))

    print_graph_summary(expr)

    mermaid = to_mermaid(expr)
    print("\n=== Mermaid DAG ===")
    print(mermaid)


if __name__ == "__main__":
    main()
