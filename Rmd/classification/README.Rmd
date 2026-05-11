# Classification — Examples

Deep-learning note
- The PyTorch MLP classifier now supports configurable hidden activation, hidden normalization, and weight initialization, in addition to the existing hidden-size and dropout controls.

These examples show how `daltoolboxdp` exposes Python classifiers, especially from the scikit-learn ecosystem, with the same training and prediction flow used in `daltoolbox`.

Use this section when you want to compare modeling behavior:

- neighborhood-based classification (`KNN`)
- probabilistic classification (`Naive Bayes`)
- margin-based classification (`SVC`)
- tree ensembles (`Random Forest`, `Gradient Boosting`)
- neural classification (`MLP`)

Scikit-learn
- [01_skcla_knn.md](01_skcla_knn.md) — K-Nearest Neighbors: predicts by majority vote among the k nearest neighbors.
- [02_skcla_nb.md](02_skcla_nb.md) — Naive Bayes: applies Bayes' rule with a conditional independence assumption among features.
- [03_skcla_svc.md](03_skcla_svc.md) — SVM (Support Vector Machine): maximizes margin; kernels enable nonlinear decision boundaries.
- [04_skcla_rf.md](04_skcla_rf.md) — Random Forest: ensemble of decision trees with majority voting; robust via bagging and feature randomness.
- [05_skcla_gb.md](05_skcla_gb.md) — Gradient Boosting: additive ensemble of trees fit to gradients of the loss.
- [06_skcla_mlp.md](06_skcla_mlp.md) — MLP (feedforward neural network): learns nonlinear decision boundaries via backpropagation.

PyTorch
- [07_torch_cla_mlp.md](07_torch_cla_mlp.md) — PyTorch MLP classifier: neural classification with configurable static/dynamic validation and stopping rules.
- [08_torch_cla_mlp_dynamic_patience.md](08_torch_cla_mlp_dynamic_patience.md) — PyTorch MLP classifier with dynamic validation and patience-based early stopping.
