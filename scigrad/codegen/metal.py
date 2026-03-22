import numpy as np

from scigrad.codegen.cpu import CPUBackend


class MetalBackend:
    """Metal backend scaffold with CPU fallback realization."""

    def __init__(self):
        try:
            import metalcompute as _  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"metalcompute not available: {e}")
        self._cpu_fallback = CPUBackend()

    def compile(self, kernel_source: str):
        return kernel_source

    def alloc(self, size_in_bytes: int):
        return np.empty((size_in_bytes,), dtype=np.uint8)

    def run(self, binary, buffers, global_size, local_size):
        raise NotImplementedError("Metal kernel execution is not implemented yet.")

    def realize(self, root_op):
        return self._cpu_fallback.realize(root_op)
