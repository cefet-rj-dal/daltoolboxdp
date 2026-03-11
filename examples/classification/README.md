# Classification — Examples

These examples show how `daltoolboxdp` exposes Python classifiers, especially from the scikit-learn ecosystem, with the same training and prediction flow used in `daltoolbox`.

Use this section when you want to compare modeling behavior:

- neighborhood-based classification (`KNN`)
- probabilistic classification (`Naive Bayes`)
- margin-based classification (`SVC`)
- tree ensembles (`Random Forest`, `Gradient Boosting`)
- neural classification (`MLP`)

- [skcla_knn.md](examples/classification/skcla_knn.md) — K-Nearest Neighbors: predicts by majority vote among the k nearest neighbors.
- [skcla_nb.md](examples/classification/skcla_nb.md) — Naive Bayes: applies Bayes’ rule with a conditional independence assumption among features.
- [skcla_svc.md](examples/classification/skcla_svc.md) — SVM (Support Vector Machine): maximizes margin; kernels enable nonlinear decision boundaries.
- [skcla_rf.md](examples/classification/skcla_rf.md) — Random Forest: ensemble of decision trees with majority voting; robust via bagging and feature randomness.
- [skcla_gb.md](examples/classification/skcla_gb.md) — Gradient Boosting: additive ensemble of trees fit to gradients of the loss.
- [skcla_mlp.md](examples/classification/skcla_mlp.md) — MLP (feedforward neural network): learns nonlinear decision boundaries via backpropagation.
