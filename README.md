<div align="center">

# scigrad

Lazy tensor computation and autodiff for scientific + ML workloads, with a NumPy-style API.

### Graph-first execution • CPU-stable today • Accelerator path in progress

[Build Plan](BUILD_PLAN.md) | [Quick Demo](examples/quickstart_demo.py) | [Tests](tests)

</div>

---

scigrad is an end-to-end experimental stack:

- **Tensor library** with reverse-mode autodiff
- **UOp-based IR** and lazy graph execution
- **Scheduler + backend abstraction**
- **NumPy-compatible frontend** for familiar workflows

The intent is to sit between high-level ergonomics and low-level hackability: easy to use, while still transparent enough to inspect how execution is built.

---

## How scigrad compares

**PyTorch**

- Similar: tensor API, autograd, straightforward training-loop style.
- Different: smaller surface area and explicit internal representation through UOps.

**tinygrad-style systems**

- Similar: lazy graph execution, compact architecture, backend-driven realization.
- Different: this project currently prioritizes correctness/stability gates before broad backend parity.

**NumPy**

- Similar: familiar namespace usage through [scigrad/numpy.py](scigrad/numpy.py).
- Different: operations build lazy graphs and compute at realization boundaries.

---

## Current status

- CPU backend is the stable baseline and passes the current test suite.
- OpenCL backend is available via environment selection and under active hardening.
- Core modules implemented: [scigrad/tensor.py](scigrad/tensor.py), [scigrad/scheduler.py](scigrad/scheduler.py), [scigrad/codegen/cpu.py](scigrad/codegen/cpu.py), [scigrad/numpy.py](scigrad/numpy.py), [scigrad/nn.py](scigrad/nn.py).
- Planned production milestones are tracked in [BUILD_PLAN.md](BUILD_PLAN.md).

---

## Installation

```bash
pip install -r requirements.txt
```

If you prefer local editable development:

```bash
pip install -e .
```

---

## Quick start

### 1) Run tests (CPU baseline)

Windows PowerShell:

```powershell
$env:SCIGRAD_BACKEND='CPU'
python -m unittest discover tests
```

macOS/Linux:

```bash
SCIGRAD_BACKEND=CPU python -m unittest discover tests
```

### 2) Run a visible demo

Windows PowerShell:

```powershell
$env:SCIGRAD_BACKEND='CPU'
$env:PYTHONPATH='.'
python examples/quickstart_demo.py
```

The demo prints:

- Elementwise tensor math
- Autodiff gradient values
- Linear layer forward-pass output

### 3) Inspect the computation graph (DAG)

Windows PowerShell:

```powershell
$env:SCIGRAD_BACKEND='CPU'
$env:PYTHONPATH='.'
python examples/show_graph.py
```

This prints:

- Graph summary (root node, node count, op list)
- Mermaid DAG text you can paste into any Mermaid viewer

---

## Minimal usage

```python
import scigrad.numpy as sgnp

a = sgnp.array([1.0, 2.0, 3.0])
b = sgnp.array([4.0, 5.0, 6.0])

out = sgnp.add(a, b)
print(out.realize()._op.inputs[0])
```

```python
x = sgnp.array([3.0])
y = sgnp.array([4.0])
loss = sgnp.add(sgnp.multiply(x, y), x)  # x*y + x
loss.backward()

print(x.grad.realize()._op.inputs[0])
print(y.grad.realize()._op.inputs[0])
```

---

## Backends

Available backend paths in the current codebase:

- CPU: [scigrad/codegen/cpu.py](scigrad/codegen/cpu.py)
- OpenCL: [scigrad/codegen/opencl.py](scigrad/codegen/opencl.py)

Select backend by environment variable:

```powershell
$env:SCIGRAD_BACKEND='CPU'    # stable baseline
# or
$env:SCIGRAD_BACKEND='OPENCL' # accelerator path
```

OpenCL mode is configured for correctness-first behavior:

- Default: OpenCL backend with safe CPU fallback for kernel execution.
- Experimental native OpenCL kernels: set `SCIGRAD_OPENCL_ALLOW_UNSAFE=1`.
- To force stable mode in PowerShell sessions where the variable may already exist:
   `Remove-Item Env:SCIGRAD_OPENCL_ALLOW_UNSAFE -ErrorAction SilentlyContinue`

Windows PowerShell example for experimental mode:

```powershell
$env:SCIGRAD_BACKEND='OPENCL'
$env:SCIGRAD_OPENCL_ALLOW_UNSAFE='1'
python -m unittest discover tests
```

---

## Project structure

```text
scigrad/
   scigrad/
      tensor.py
      scheduler.py
      device.py
      numpy.py
      nn.py
      codegen/
         cpu.py
         opencl.py
   examples/
      quickstart_demo.py
   tests/
      test_layer1.py
      test_layer2.py
      test_layer3.py
      test_layer4.py
      test_layer5.py
      test_layer6.py
```

---

## Contributing

Contributions should focus on correctness, test coverage, and clarity.

- Add regression tests for every bug fix.
- Keep diffs scoped and easy to review.
- Prefer readable implementations over clever shortcuts.
- Validate with CPU baseline tests before backend-specific changes.

Run tests before opening a PR:

```bash
python -m unittest discover tests
```

---

## Roadmap

The active roadmap includes:

- Scheduler fusion verification coverage
- OpenCL correctness parity against CPU
- Additional primitive coverage and gradient rules
- Expanded NumPy compatibility
- Broader NN layers and optimizer stack

See details in [BUILD_PLAN.md](BUILD_PLAN.md).
