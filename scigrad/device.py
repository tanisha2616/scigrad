# This file handles device detection and backend selection.
import os
from scigrad.codegen.cpu import CPUBackend

# Default to CPU backend
Device = CPUBackend()

# Attempt to use OpenCL if available and not overridden
if os.environ.get("SCIGRAD_BACKEND") != "CPU":
    try:
        from scigrad.codegen.opencl import OpenCLBackend
        Device = OpenCLBackend()
        print("scigrad: Using OpenCL backend.")
    except Exception as e:
        print(f"scigrad: OpenCL backend failed to load ({e}), falling back to CPU.")
        Device = CPUBackend()
