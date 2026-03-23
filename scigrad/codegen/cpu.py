import numpy as np
from typing import Dict, Any, Optional
from scigrad.tensor import UOp
from scigrad.kernel import KernelSpec

class CPUBackend:
    """
    Fallback backend that executes kernels using NumPy.
    It is not allowed to be slower than plain NumPy.
    """
    def __init__(self):
        self._cache: Dict[UOp, np.ndarray] = {}

    def run(self, kernel: KernelSpec, input_refcount: Optional[Dict[UOp, int]] = None):
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
                    if inp_node in local_cache:
                        inputs.append(local_cache[inp_node])
                    elif inp_node in self._cache:
                        inputs.append(self._cache[inp_node])
                    else:
                        raise RuntimeError(f"Intermediate input {inp_node} not found in caches!")
                else: # It's a raw np.ndarray from a CONST op
                    inputs.append(inp_node)

            if input_refcount is not None:
                for inp_node in node.inputs:
                    if isinstance(inp_node, UOp) and inp_node.op != 'LOAD':
                        remaining = input_refcount.get(inp_node, 0)
                        if remaining > 0:
                            remaining -= 1
                            input_refcount[inp_node] = remaining
                            if remaining == 0 and inp_node in self._cache:
                                del self._cache[inp_node]

            # Execute the operation
            if node.op == 'CONST':
                result = inputs[0]
            elif node.op == 'NEG':
                result = -inputs[0]
            elif node.op == 'EXP':
                result = np.exp(inputs[0])
            elif node.op == 'LOG':
                result = np.log(inputs[0])
            elif node.op == 'SQRT':
                result = np.sqrt(inputs[0])
            elif node.op == 'SIN':
                result = np.sin(inputs[0])
            elif node.op == 'COS':
                result = np.cos(inputs[0])
            elif node.op == 'ABS':
                result = np.abs(inputs[0])
            elif node.op == 'SIGN':
                result = np.sign(inputs[0])
            elif node.op == 'ADD':
                result = inputs[0] + inputs[1]
            elif node.op == 'MUL':
                result = inputs[0] * inputs[1]
            elif node.op == 'ASSIGN':
                result = inputs[1]
            elif node.op == 'MAX':
                result = np.maximum(inputs[0], inputs[1])
            elif node.op == 'CMPEQ':
                result = (inputs[0] == inputs[1]).astype(inputs[0].dtype)
            elif node.op == 'CMPLT':
                result = (inputs[0] < inputs[1]).astype(inputs[0].dtype)
            elif node.op == 'WHERE':
                result = np.where(inputs[0] != 0, inputs[1], inputs[2])
            elif node.op in ('TRANSPOSE', 'PERMUTE'):
                result = inputs[0].transpose(node.args[0])
            elif node.op == 'MATMUL':
                result = inputs[0] @ inputs[1]
            elif node.op == 'RECIP':
                result = 1 / inputs[0]
            elif node.op == 'RESHAPE':
                result = inputs[0].reshape(node.args[0])
            elif node.op == 'PAD':
                result = np.pad(inputs[0], node.args[0], mode='constant')
            elif node.op == 'SHRINK':
                slices = tuple(slice(start, end) for start, end in node.args[0])
                result = inputs[0][slices]
            elif node.op == 'EXPAND':
                result = np.broadcast_to(inputs[0], node.args[0])
            elif node.op == 'CAST':
                result = inputs[0].astype(node.args[0])
            elif node.op == 'REDUCE_SUM':
                axis = node.args[0]
                if axis is None:
                    res = inputs[0].sum()
                else:
                    res = inputs[0].sum(axis=axis, keepdims=node.args[1])
                
                if res.shape != node.shape:
                    res = res.reshape(node.shape)
                result = res
            elif node.op == 'REDUCE_MAX':
                axis = node.args[0]
                if axis is None:
                    res = inputs[0].max()
                else:
                    res = inputs[0].max(axis=axis, keepdims=node.args[1])

                if res.shape != node.shape:
                    res = res.reshape(node.shape)
                result = res
            elif node.op == 'REDUCE_PROD':
                axis = node.args[0]
                if axis is None:
                    res = inputs[0].prod()
                else:
                    res = inputs[0].prod(axis=axis, keepdims=node.args[1])

                if res.shape != node.shape:
                    res = res.reshape(node.shape)
                result = res
            else:
                raise NotImplementedError(f"CPUBackend does not support op '{node.op}'")
            
            local_cache[node] = result

        output_uop = kernel.uops[-1]
        final_result = local_cache[output_uop]

        for node, arr in local_cache.items():
            if node.op == 'LOAD':
                continue
            if node is output_uop or (input_refcount is not None and input_refcount.get(node, 0) > 0):
                self._cache[node] = arr

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
        visited: set[UOp] = set()

        def populate_loads(node: UOp):
            if node in visited:
                return
            visited.add(node)

            if node.op in ('LOAD', 'CONST'):
                # For CONST, the input is the numpy array itself
                self._cache[node] = node.inputs[0]

            for inp in node.inputs:
                if isinstance(inp, UOp):
                    populate_loads(inp)
        populate_loads(root_op)
        
        if not kernels:
            # The root op was a LOAD or CONST, so no kernels were generated.
            return self._cache[root_op]

        input_refcount: Dict[UOp, int] = {}
        for k in kernels:
            for node in k.uops:
                for inp in node.inputs:
                    if isinstance(inp, UOp) and inp.op != 'LOAD':
                        input_refcount[inp] = input_refcount.get(inp, 0) + 1

        final_result = None
        for k in kernels:
            final_result = self.run(k, input_refcount=input_refcount)
            
        return final_result
