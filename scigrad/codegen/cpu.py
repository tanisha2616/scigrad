import numpy as np
from typing import Dict, Any
from scigrad.tensor import UOp
from scigrad.kernel import KernelSpec

class CPUBackend:
    """
    Fallback backend that executes kernels using NumPy.
    It is not allowed to be slower than plain NumPy.
    """
    def __init__(self):
        self._cache: Dict[UOp, np.ndarray] = {}

    def run(self, kernel: KernelSpec):
        """
        Executes a KernelSpec. For the CPU backend, this means interpreting
        the UOps and calling the corresponding NumPy functions.
        """
        # The cache for this kernel execution only
        local_cache: Dict[UOp, np.ndarray] = {}

        # Populate local_cache with inputs for this kernel from the global cache
        # The new KernelSpec has an 'inputs' dict {uid: data}
        for uop in kernel.uops:
            for inp in uop.inputs:
                 if isinstance(inp, UOp) and inp.op == "LOAD":
                    if inp in self._cache:
                        local_cache[inp] = self._cache[inp]


        # Execute the ops in the kernel in order
        for node in kernel.uops:
            inputs = []
            for inp_node in node.inputs:
                if isinstance(inp_node, UOp):
                    if inp_node not in local_cache:
                        raise RuntimeError(f"Intermediate input {inp_node} not found in local cache!")
                    inputs.append(local_cache[inp_node])
                else: # It's a raw np.ndarray from a CONST op
                    inputs.append(inp_node)

            # Execute the operation
            if node.op == 'NEG':
                result = -inputs[0]
            elif node.op == 'ADD':
                result = inputs[0] + inputs[1]
            elif node.op == 'MUL':
                result = inputs[0] * inputs[1]
            elif node.op == 'TRANSPOSE':
                result = inputs[0].transpose(node.args[0])
            elif node.op == 'MATMUL':
                result = inputs[0] @ inputs[1]
            elif node.op == 'RECIP':
                result = 1 / inputs[0]
            elif node.op == 'RESHAPE':
                result = inputs[0].reshape(node.args[0])
            elif node.op == 'EXPAND':
                result = np.broadcast_to(inputs[0], node.args[0])
            elif node.op == 'REDUCE_SUM':
                axis = node.args[0]
                if axis is None:
                    res = inputs[0].sum()
                else:
                    res = inputs[0].sum(axis=axis, keepdims=node.args[1])
                
                if res.shape != node.shape:
                    res = res.reshape(node.shape)
                result = res
            else:
                raise NotImplementedError(f"CPUBackend does not support op '{node.op}'")
            
            local_cache[node] = result
        
        # The result of the kernel is the result of its final op
        final_result = local_cache[kernel.uops[-1]]
        # Store the final result in the global cache
        self._cache[kernel.uops[-1]] = final_result
        return final_result

    def realize(self, root_op: UOp) -> np.ndarray:
        """
        Schedules and executes the graph ending at the given root UOp.
        """
        from scigrad.scheduler import schedule
        
        # The scheduler gives us an ordered list of kernels to run.
        kernels = schedule(root_op)
        
        # Clear global cache for this realization pass
        self._cache.clear()

        # Pre-populate the cache with any LOAD or CONST ops.
        def populate_loads(node: UOp):
            if node.op in ('LOAD', 'CONST'):
                # For CONST, the input is the numpy array itself
                self._cache[node] = node.inputs[0]
            for inp in node.inputs:
                if isinstance(inp, UOp):
                    if inp not in self._cache:
                         populate_loads(inp)
        populate_loads(root_op)
        
        if not kernels:
            # The root op was a LOAD or CONST, so no kernels were generated.
            return self._cache[root_op]

        final_result = None
        for k in kernels:
            final_result = self.run(k)
            
        return final_result
