import numpy as np
import pyopencl as cl
import hashlib
import os
from typing import Dict, List, Tuple, Any

from scigrad.tensor import UOp
from scigrad.kernel import KernelSpec
from scigrad.scheduler import schedule

# Mapping from UOp op names to OpenCL C function names or expressions
OCL_OP_MAP = {
    "NEG": "-",
    "ADD": "+",
    "MUL": "*",
    "RECIP": "1.0f/",
    "TRANSPOSE": "transpose", # This will be a custom function
    "MATMUL": "matmul",       # This will be a custom function
}

def codegen_opencl(kernel: KernelSpec) -> str:
    """
    Generates OpenCL C code from a KernelSpec.
    """
    op_list = kernel.uops
    
    # Header
    code_lines = []
    
    # Kernel signature
    input_names = sorted(kernel.inputs.keys())
    kernel_args = [f"__global const float* in{i}" for i in range(len(input_names))]
    kernel_args.append("__global float* out")
    code_lines.append(f"__kernel void scigrad_kernel({', '.join(kernel_args)}) {{")
    
    # Kernel body
    code_lines.append("    int gid = get_global_id(0);")
    
    uop_to_var = {}
    
    # Pre-map all UOps to their C variable names. This is the crucial fix.
    # LOAD ops are mapped to input buffers, others to intermediate variables.
    for i, uid in enumerate(input_names):
        # Find the corresponding UOp in the op_list for the input uid
        for uop in op_list:
            if uop.op == "LOAD" and uop.uid == uid:
                uop_to_var[uop] = f"in{i}"
                break
    
    for uop in op_list:
        if uop.op != "LOAD":
            uop_to_var[uop] = f"v{uop.uid}"


    # Generate code for each operation
    for uop in op_list:
        if uop.op == "LOAD":
            continue

        # Get the C code for the inputs
        input_vars = []
        for i_uop in uop.inputs:
            var = uop_to_var[i_uop]
            # If the input is a buffer and the current op is element-wise, access with [gid]
            if var.startswith("in") and uop.op not in ('REDUCE_SUM', 'MATMUL'):
                 input_vars.append(f"{var}[gid]")
            else:
                input_vars.append(var)

        output_var = uop_to_var[uop]

        if uop.op in OCL_OP_MAP:
            if uop.op == "MATMUL":
                a, b = uop.inputs
                M, K = a.shape
                _K, N = b.shape
                a_var, b_var = uop_to_var[a], uop_to_var[b]
                
                code_lines.append(f"    float {output_var} = 0.0f;")
                code_lines.append(f"    int row = gid / {N};")
                code_lines.append(f"    int col = gid % {N};")
                code_lines.append(f"    if (row < {M} && col < {N}) {{")
                code_lines.append(f"        for (int k = 0; k < {K}; ++k) {{")
                code_lines.append(f"            {output_var} += {a_var}[row * {K} + k] * {b_var}[k * {N} + col];")
                code_lines.append(f"        }}")
                code_lines.append(f"    }}")
                continue 

            elif uop.op == "TRANSPOSE":
                 raise NotImplementedError("OpenCL codegen for op 'TRANSPOSE' not implemented.")

            if len(input_vars) == 1:
                line = f"float {output_var} = {OCL_OP_MAP[uop.op]}({input_vars[0]});"
            else:
                line = f"float {output_var} = {input_vars[0]} {OCL_OP_MAP[uop.op]} {input_vars[1]};"
            code_lines.append(f"    {line}")

        elif uop.op == 'REDUCE_SUM':
            input_var = uop_to_var[uop.inputs[0]]
            input_size = np.prod(uop.inputs[0].shape)
            code_lines.append(f"    float {output_var} = 0.0f;")
            code_lines.append(f"    if (gid == 0) {{")
            code_lines.append(f"        for(int i=0; i<{input_size}; ++i) {{ {output_var} += {input_var}[i]; }}")
            code_lines.append(f"    }}")

        elif uop.op == 'EXPAND':
            original_size = np.prod(uop.inputs[0].shape)
            input_var_base = uop_to_var[uop.inputs[0]]
            line = f"float {output_var} = {input_var_base}[gid % {original_size}];"
            code_lines.append(f"    {line}")

    # Write the final result to the output buffer
    final_uop = op_list[-1]
    final_var = uop_to_var[final_uop]
    
    if final_uop.op == 'REDUCE_SUM':
        code_lines.append(f"    if (gid == 0) {{ out[0] = {final_var}; }}")
    elif final_uop.op == 'MATMUL':
         M, N = final_uop.shape
         code_lines.append(f"    if (gid < {M*N}) {{ out[gid] = {final_var}; }}")
    else:
         code_lines.append(f"    out[gid] = {final_var};")
    
    code_lines.append("}")
    return " ".join(code_lines)

class OpenCLBackend:
    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self._cache: Dict[UOp, cl.Buffer] = {}
        self._prog_cache: Dict[str, cl.Program] = {}

    def run(self, kernel: KernelSpec, output_shape: Tuple[int, ...]):
        
        # Generate OpenCL code for the kernel
        code = codegen_opencl(kernel)
        
        # Cache the compiled program
        if code not in self._prog_cache:
            self._prog_cache[code] = cl.Program(self.ctx, code).build()
        
        prg = self._prog_cache[code]

        # Create OpenCL buffers for the inputs
        input_buffers: List[cl.Buffer] = []
        
        # The order of inputs to the OpenCL kernel is determined by the sorted UIDs
        sorted_input_uids = sorted(kernel.inputs.keys())
        
        for uid in sorted_input_uids:
            data = kernel.inputs[uid]
            buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.astype(np.float32))
            input_buffers.append(buf)

        # Create the output buffer
        output_buffer = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=np.prod(output_shape) * 4)
        
        # Execute the kernel
        kernel_args = tuple(input_buffers) + (output_buffer,)
        prg.scigrad_kernel(self.queue, (np.prod(output_shape),), None, *kernel_args)

        # Read the result back from the output buffer
        result = np.empty(output_shape, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, output_buffer).wait()
        
        return result

    def realize(self, root_op: UOp) -> np.ndarray:
        """
        Schedules and executes the graph ending at the given root UOp.
        """
        kernels = schedule(root_op)
        
        self._cache.clear()

        def populate_loads(node: UOp):
            if node.op in ('LOAD', 'CONST'):
                # For CONST, the input is the numpy array itself
                # For LOAD, it's the first input
                # This part of the logic is tricky for OpenCL as it involves host->device transfer.
                # Let's assume for now that the data is available when the kernel needs it.
                pass
            for inp in node.inputs:
                if isinstance(inp, UOp):
                    populate_loads(inp)
        populate_loads(root_op)
        
        if not kernels:
            # The root op was a LOAD or CONST, so no kernels were generated.
            # We need to get the data from the op itself.
            # This case needs careful handling for OpenCL.
            if root_op.op in ('LOAD', 'CONST'):
                return root_op.inputs[0]
            else:
                # This should not happen if the scheduler is correct
                raise ValueError("No kernels generated for a non-load/const op")

        final_result = None
        for k in kernels:
            # The output shape of the kernel is the shape of its last op
            output_shape = k.uops[-1].shape
            final_result = self.run(k, output_shape)
            
        return final_result