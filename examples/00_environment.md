# Environment Check — `reticulate`, Python, and GPU

Use this page before running the PyTorch-backed examples in `daltoolboxdp`.

Goals
- choose which Python environment `reticulate` will use
- verify that PyTorch is visible from R
- verify whether the backend will run on GPU or CPU

## 1. Choose the Python environment

Select the environment before loading `daltoolboxdp`.

```r
library(reticulate)

use_virtualenv("/opt/venv/dal", required = TRUE)
# or:
# use_python("/opt/venv/dal/bin/python", required = TRUE)

py_config()
```

Alternative from the shell before starting R:

```bash
export RETICULATE_PYTHON=/opt/venv/dal/bin/python
```

## 2. Check Python and Torch from R

```r
library(reticulate)

py_run_string("
import sys
import torch
print('python:', sys.executable)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', 'cuda:0' if torch.cuda.is_available() else 'cpu')
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
")
```

Interpretation
- If `cuda available: True`, the PyTorch backend can use GPU.
- If `cuda available: False`, the PyTorch backend will run on CPU.

## 3. Force CPU or choose a GPU

Set this before Python is initialized by `reticulate`.

Force CPU:

```r
Sys.setenv(CUDA_VISIBLE_DEVICES = "")
```

Use GPU 0:

```r
Sys.setenv(CUDA_VISIBLE_DEVICES = "0")
```

## 4. Monitor CPU and GPU while the example runs

CPU:

```bash
htop
```

GPU:

```bash
nvtop
```

Interpretation
- In `htop`, watch the Python process started by `reticulate`.
- In `nvtop`, watch GPU utilization, GPU memory usage, and the Python process using the device.
- If the model is using GPU, GPU utilization or memory usage should increase during training.

Fallback if `nvtop` is not installed:

```bash
watch -n 1 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

## 5. Minimal DAL test with PyTorch on GPU

This test tries to run the PyTorch-backed classifier on GPU.

```r
library(reticulate)
Sys.setenv(CUDA_VISIBLE_DEVICES = "0")
use_virtualenv("/opt/venv/dal", required = TRUE)
py_config()

library(daltoolbox)
library(daltoolboxdp)

iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

model <- torch_cla_mlp(
  attribute = "Species",
  slevels = slevels,
  input_size = 4L,
  hidden_sizes = c(16L, 8L),
  num_classes = 3L,
  epochs = 100L
)

model <- fit(model, sr$train)
pred <- predict(model, sr$test)
evaluate(model, sr$test[, "Species"], pred)$metrics
```

## 6. Minimal DAL test with PyTorch on CPU

This test forces the same example to run on CPU.

Start a fresh R session before running this block, so `reticulate` does not reuse an already initialized Python process.

```r
library(reticulate)
Sys.setenv(CUDA_VISIBLE_DEVICES = "")
use_virtualenv("/opt/venv/dal", required = TRUE)
py_config()

library(daltoolbox)
library(daltoolboxdp)

iris <- datasets::iris
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)

model <- torch_cla_mlp(
  attribute = "Species",
  slevels = slevels,
  input_size = 4L,
  hidden_sizes = c(16L, 8L),
  num_classes = 3L,
  epochs = 100L
)

model <- fit(model, sr$train)
pred <- predict(model, sr$test)
evaluate(model, sr$test[, "Species"], pred)$metrics
```

## 7. How `daltoolboxdp` chooses GPU or CPU

The PyTorch wrappers in `daltoolboxdp` choose the device automatically:

```python
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

So the effective device depends on:
- the Python environment selected by `reticulate`
- whether that environment has a CUDA-capable PyTorch build
- whether the NVIDIA driver is available on the host

## 8. Recommended next example

After this check, start with:

- [classification/07_torch_cla_mlp.md](classification/07_torch_cla_mlp.md)
