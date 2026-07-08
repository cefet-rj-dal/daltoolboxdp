# GPU — Environment And Device Checks

Use this section before running the PyTorch-backed examples in `daltoolboxdp`. The goal is to make the Python environment selected by `reticulate` explicit and to confirm whether PyTorch will execute on GPU or CPU.

This page helps answer:

- which Python interpreter `reticulate` is bound to
- whether the selected environment exposes a CUDA-enabled PyTorch build
- how to force CPU execution or restrict execution to a specific GPU
- how to monitor device usage while DAL examples are running

- [00_environment.md](00_environment.md) — Select the Python environment used by `reticulate`, verify PyTorch visibility, and confirm whether the backend will run on GPU or CPU.
