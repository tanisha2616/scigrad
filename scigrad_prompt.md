You are building scigrad: a lazy tensor computation framework that works as a drop-in for both NumPy-based scientific computing (AstroPy, BioPython, genomics pipelines) and ML workloads (neural networks, autodiff, training loops). It runs on any hardware that exposes OpenCL, Metal, or CUDA — including integrated GPUs on i3 laptops — and falls back to NumPy silently if no accelerator is found.

The design is tinygrad's architecture extended to cover scientific primitives. Same lazy DAG. Same kernel fusion scheduler. Same codegen abstraction. Wider op set. The user imports scigrad, uses it like NumPy, calls .realize() at the boundary, and gets results. On a GPU-less i3, the iGPU runs OpenCL. On a Mac, Metal. On a workstation, CUDA. The code they wrote is identical in all three cases.

---

Core principle

Every tensor operation returns a new lazy Tensor backed by a UOp node. No computation happens at call time. The graph builds. When .realize() is called — or when the user accesses .numpy() or prints the tensor — the scheduler walks the graph, fuses what can be fused, emits the minimum set of device kernels, compiles them, runs them, and returns concrete data. The user never manages this. They write computation. The framework decides execution.

---

Layer 1 — Tensor and UOp

Tensor wraps either a concrete numpy array or a UOp. It must expose the full NumPy array interface — shape, dtype, ndim, __add__, __mul__, __sub__, __truediv__, __neg__, __matmul__, __getitem__, T, reshape, squeeze, unsqueeze, broadcast_to. Every one of these records a UOp rather than executing.

UOp is an immutable dataclass: op (string enum), inputs (tuple of UOp or ndarray), shape (tuple of ints), dtype (string). Shape and dtype are inferred at construction time from the inputs. If a shape cannot be statically determined, raise immediately with the op name and input shapes in the message.

The primitive op set — everything must decompose into these and only these:

Memory: LOAD, STORE, CONST
Elementwise unary: NEG, EXP, LOG, SQRT, SIN, COS, ABS, SIGN, RECIP
Elementwise binary: ADD, MUL, MAX, CMPEQ, CMPLT, WHERE
Type: CAST
Shape: RESHAPE, PERMUTE, EXPAND, SHRINK, PAD
Reduction: REDUCE_SUM, REDUCE_MAX, REDUCE_PROD
Special: CONTIGUOUS, ASSIGN

This is 27 primitives. Everything else is sugar. Subtraction is NEG then ADD. Division is RECIP then MUL. Softmax is EXP divided by its REDUCE_SUM. Convolution is EXPAND, SHRINK, MUL, REDUCE_SUM. A Voigt profile is CONST arithmetic on a frequency axis. A cross-correlation is two FFTs, a complex MUL, and an inverse FFT — where FFT is itself decomposed into butterfly stages using the primitives above or realized as a single LOAD/STORE pair wrapping a cuFFT/clFFT call depending on backend.

---

Layer 2 — Autodiff

Autodiff is implemented as a backward pass over the UOp DAG. Every primitive op must have a registered gradient rule that expresses the gradient in terms of the same primitives. Gradient accumulation uses REDUCE_SUM over broadcast dimensions.

.backward() walks the graph in reverse topological order, calls each op's gradient function with the upstream gradient, accumulates into .grad on each Tensor. No separate tape or context object. The forward graph IS the tape.

The autodiff rules required at minimum: ADD (identity both ways), MUL (swap inputs), NEG (negate), EXP (multiply by forward output), LOG (multiply by RECIP of input), SQRT (divide by 2*forward output), REDUCE_SUM (EXPAND upstream grad to input shape), RESHAPE/PERMUTE/EXPAND (reverse the shape op), SHRINK/PAD (reverse with zeros).

---

Layer 3 — Scheduler

The scheduler takes the UOp DAG rooted at one or more output tensors and returns an ordered list of KernelSpec objects. Each KernelSpec describes exactly one device kernel.

Fusion rules:

Any two adjacent elementwise ops fuse. Elementwise means every output element depends on exactly one input element at the same position, after broadcasting.

RESHAPE and PERMUTE fuse into adjacent elementwise ops if they do not require a data copy (i.e. the result is contiguous or the access pattern is expressible as a strided index calculation).

REDUCE_SUM and REDUCE_MAX terminate a fusion group. The reduction itself is its own KernelSpec. Ops that depend on the reduction output start a new group.

CONTIGUOUS forces materialization — it is a group boundary in both directions.

A KernelSpec contains: a list of UOps in execution order, input buffer references, output buffer references, the reduction axis and type if applicable, the global work size (number of output elements), and the local work size (chosen by the scheduler based on device query).

---

Layer 4 — Backends

Each backend implements three functions: compile(kernel_source) -> binary, run(binary, buffers, global_size, local_size), and alloc(size_in_bytes) -> buffer. Nothing else. The scheduler and codegen are backend-agnostic.

OpenCL backend uses pyopencl. Device selection prefers GPU over CPU, prefers discrete over integrated, but falls back down the list rather than failing. Compiled binaries are cached on disk keyed by SHA256 of kernel source. Cache lives in the OS temp directory under a fixed subdirectory. Cache is valid across process restarts.

Metal backend uses metalcompute if available. Same interface.

CUDA backend uses pycuda if available. Same interface.

CPU fallback backend lowers every KernelSpec directly to NumPy operations. It must produce bit-identical results to the GPU backends for float32 inputs. It is not allowed to be slower than plain NumPy for equivalent computation.

Backend selection order: CUDA if nvidia device found, Metal if on macOS with GPU, OpenCL if any OpenCL device found including iGPU, CPU fallback last. Selection happens once at import time, stored as module-level state. User can override via environment variable SCIGRAD_BACKEND.

---

Layer 5 — Codegen

Codegen takes a KernelSpec and emits a kernel source string for the target backend language (OpenCL C, Metal Shading Language, or CUDA C).

For elementwise KernelSpecs: emit a single kernel where each thread handles one output element. The body is the inlined expression tree of the fused UOps, written as a single C expression where possible. Intermediate values that appear more than once in the expression tree are assigned to local variables.

For reduction KernelSpecs: emit a two-pass reduction. First pass reduces blocks into partial results using local memory and synchronization. Second pass reduces partial results. This is standard — implement it correctly, not cleverly.

Index arithmetic for strided/permuted/broadcast accesses is emitted as explicit integer arithmetic in the kernel. No pointer aliasing. No implicit broadcasting in the kernel — every access pattern is explicit.

---

Layer 6 — NumPy compatibility layer

This is the layer that makes existing AstroPy and BioPython code run on scigrad without modification.

Implement a module scigrad.numpy that mirrors the numpy namespace. ndarray is aliased to Tensor. All numpy functions that AstroPy and BioPython actually call — zeros, ones, array, linspace, arange, concatenate, stack, dot, einsum, fft.fft, fft.ifft, fft.fftfreq, where, clip, abs, sqrt, exp, log, sin, cos, sum, mean, std, var, argmax, argmin, argsort, searchsorted, interp — are implemented in terms of Tensor operations.

The compatibility shim is used like this:

    import scigrad.numpy as np

Everything after that line works as before, but is now lazy and GPU-accelerated.

Correctness requirement: AstroPy's Spectrum1D operations, when run through scigrad.numpy as the backend, must produce results within float32 rounding error of the same operations run through standard NumPy. This is the acceptance test for the compatibility layer.

---

Layer 7 — Neural network ops

Implement scigrad.nn with: Linear, Conv2d, BatchNorm2d, LayerNorm, MultiheadAttention, Embedding, Dropout. These are implemented entirely using Tensor primitives — no special casing in the backend. If the primitive set is correct, neural network ops fall out naturally.

Optimizer implementations: SGD with momentum, Adam, AdamW. These update .data on leaf tensors in-place using ASSIGN.

The test for this layer: train a small transformer (2 layers, 64 hidden dim) on character-level text prediction. Loss must decrease. It does not need to be fast — it needs to be correct.

---

File structure

scigrad/tensor.py — Tensor class, UOp dataclass, all operator overloads
scigrad/ops.py — primitive op definitions and their gradient rules
scigrad/scheduler.py — fusion logic, KernelSpec
scigrad/codegen/ — one file per backend: opencl.py, metal.py, cuda.py, cpu.py
scigrad/numpy.py — NumPy compatibility namespace
scigrad/nn.py — neural network module and optimizers
scigrad/device.py — device detection and backend selection
tests/ — one test file per layer, run in order

---

LOC budget

tensor.py: 150 lines
ops.py: 120 lines
scheduler.py: 100 lines
codegen/opencl.py: 120 lines
codegen/cpu.py: 60 lines
codegen/metal.py and cuda.py: 80 lines each if implemented
numpy.py: 100 lines
nn.py: 150 lines
device.py: 40 lines

Total target: under 1000 lines excluding tests. If any file exceeds its budget, the abstraction is wrong. Refactor before proceeding to the next layer.

---

Build order and verification gates

Gate 1: Tensor + UOp construction. Verify that a chain of five operations produces a DAG with five nodes and zero computation. Verify shape inference is exact on reshape, broadcast, and matmul.

Gate 2: CPU codegen + scheduler. Verify that the scheduler fuses three elementwise ops into one KernelSpec. Verify the CPU backend produces NumPy-identical results for ADD, MUL, REDUCE_SUM, PERMUTE.

Gate 3: OpenCL codegen. Verify the same KernelSpecs from gate 2 run on an OpenCL device and produce the same results. If no OpenCL device is available, run the OpenCL C source through a CPU OpenCL implementation (Intel OpenCL SDK or Pocl) for correctness verification.

Gate 4: Autodiff. Verify gradients of EXP, LOG, MUL chain, REDUCE_SUM using numerical differentiation. Tolerance 1e-4.

Gate 5: NumPy compatibility. Run AstroPy's Spectrum1D with scigrad.numpy as backend. Assert outputs match reference NumPy run within float32 epsilon.

Gate 6: Neural network. Train the small transformer. Assert loss at step 100 is less than loss at step 0.

Do not proceed past a gate until the gate passes. Do not write layer N+1 until layer N is verified.

---

The benchmark that defines the project complete

Pipeline: load a float32 spectrum of 2,000,000 points, subtract a running median (window 101), cross-correlate against a template of 500 points using FFT, find the peak. Run this on Intel Iris Xe via OpenCL. Time from array construction to peak index must be under 3 seconds. NumPy CPU on the same machine must produce the same peak index. This benchmark is the first line of the README.
