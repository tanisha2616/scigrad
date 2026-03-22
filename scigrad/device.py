import os
from typing import Optional

from scigrad.codegen.cpu import CPUBackend


def _try_cuda() -> Optional[object]:
    try:
        from scigrad.codegen.cuda import CUDABackend
        return CUDABackend()
    except Exception:
        return None


def _try_metal() -> Optional[object]:
    try:
        from scigrad.codegen.metal import MetalBackend
        return MetalBackend()
    except Exception:
        return None


def _try_opencl() -> Optional[object]:
    try:
        from scigrad.codegen.opencl import OpenCLBackend
        return OpenCLBackend()
    except Exception:
        return None


def _select_backend() -> object:
    override = os.environ.get("SCIGRAD_BACKEND", "AUTO").upper()

    if override == "CPU":
        return CPUBackend()

    if override == "CUDA":
        backend = _try_cuda()
        if backend is not None:
            print("scigrad: Using CUDA backend.")
            return backend
        print("scigrad: CUDA backend unavailable, falling back to CPU.")
        return CPUBackend()

    if override == "METAL":
        backend = _try_metal()
        if backend is not None:
            print("scigrad: Using Metal backend.")
            return backend
        print("scigrad: Metal backend unavailable, falling back to CPU.")
        return CPUBackend()

    if override == "OPENCL":
        backend = _try_opencl()
        if backend is not None:
            print("scigrad: Using OpenCL backend.")
            return backend
        print("scigrad: OpenCL backend unavailable, falling back to CPU.")
        return CPUBackend()

    # AUTO selection order: CUDA -> METAL -> OPENCL -> CPU
    backend = _try_cuda()
    if backend is not None:
        print("scigrad: Using CUDA backend (auto).")
        return backend

    backend = _try_metal()
    if backend is not None:
        print("scigrad: Using Metal backend (auto).")
        return backend

    backend = _try_opencl()
    if backend is not None:
        print("scigrad: Using OpenCL backend (auto).")
        return backend

    print("scigrad: Using CPU backend (auto fallback).")
    return CPUBackend()


Device = _select_backend()
