# Regression — Examples

Deep-learning note
- The PyTorch MLP regressor now supports configurable hidden activation, output activation, hidden normalization, and weight initialization, beyond hidden-size and dropout tuning.
- It now also follows the same regression contract as `daltoolbox`: predictor columns are registered during `fit()`, optional preprocessing is fitted on those predictors, and `input_size` can be inferred from the training data.

These examples show Python-backed regression wrappers integrated into the `daltoolbox` workflow. The main use case here is numeric prediction with the same `fit` / `predict` pattern used by DAL learners.

PyTorch
- [01_torch_reg_mlp.md](01_torch_reg_mlp.md) — Base PyTorch MLP regressor example for numeric prediction.
- [02_torch_reg_mlp_static_patience.md](02_torch_reg_mlp_static_patience.md) — PyTorch MLP regressor with static validation and patience-based early stopping.
- [03_torch_reg_mlp_dynamic_patience.md](03_torch_reg_mlp_dynamic_patience.md) — PyTorch MLP regressor with dynamic validation and patience-based early stopping.
