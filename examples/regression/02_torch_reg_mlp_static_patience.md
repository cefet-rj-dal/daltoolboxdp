## 02. PyTorch MLP Regressor with Static Validation and Patience

This example emphasizes the training regime of the PyTorch-backed MLP regressor. The architecture is kept simple while validation is held fixed across epochs and early stopping is controlled by patience.

Prerequisites
- R packages: daltoolbox, daltoolboxdp
- Python with PyTorch accessible via reticulate


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# installation
#install.packages("daltoolboxdp")

library(daltoolbox)
library(daltoolboxdp)
library(MASS)
```


``` r
# Dataset for regression analysis
data(Boston)
Boston <- as.matrix(Boston)
```


``` r
# Train/test split
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test
```


``` r
# Static validation with patience-based early stopping
model <- torch_reg_mlp(
  attribute = "medv",
  input_size = ncol(Boston) - 1L,
  hidden_sizes = c(16L, 8L),
  epochs = 300L,
  validation_strategy = "static",
  stopping_rule = "patience",
  patience = 20L,
  val_ratio = 0.2
)
```

```
## Warning: restarting interrupted promise evaluation
```

```
## Warning: internal error 1 in R_decompress1 with libdeflate
```

```
## Error:
## ! lazy-load database 'C:/R/R-4.5.0/library/daltoolboxdp/R/daltoolboxdp.rdb' is corrupt
```

``` r
set_example_seed()
model <- fit(model, boston_train)
```

```
## Error:
## ! object 'model' not found
```

Training configuration
- `validation_strategy = "static"` keeps the same validation partition during the whole training run.
- `stopping_rule = "patience"` stops training when the validation loss stops improving for a chosen number of epochs.
- `epochs = 300L` acts as an upper bound; the effective number of epochs is determined by early stopping.


``` r
# Training evaluation
train_prediction <- predict(model, boston_train)
```

```
## Error:
## ! object 'model' not found
```

``` r
boston_train_predictand <- boston_train[, "medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
```

```
## Error:
## ! object 'model' not found
```

``` r
print(train_eval$metrics)
```

```
## Error:
## ! object 'train_eval' not found
```


``` r
# Test evaluation
test_prediction <- predict(model, boston_test)
```

```
## Error:
## ! object 'model' not found
```

``` r
boston_test_predictand <- boston_test[, "medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
```

```
## Error:
## ! object 'model' not found
```

``` r
print(test_eval$metrics)
```

```
## Error:
## ! object 'test_eval' not found
```


``` r
# Effective training duration
print(model$epochs_done)
```

```
## Error:
## ! object 'model' not found
```


``` r
# Training and validation curves
fit_loss <- data.frame(
  x = seq_along(model$train_loss_hist),
  train_loss = model$train_loss_hist
)
```

```
## Error:
## ! object 'model' not found
```

``` r
if (!is.null(model$val_loss_hist) && length(model$val_loss_hist) > 0) {
  fit_loss$val_loss <- model$val_loss_hist
}
```

```
## Error:
## ! object 'model' not found
```

``` r
colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
```

```
## Error:
## ! object 'fit_loss' not found
```

``` r
grf <- plot_series(fit_loss, colors = colors)
```

```
## Error:
## ! object 'fit_loss' not found
```

``` r
plot(grf)
```

```
## Error:
## ! object 'grf' not found
```

Notes
- This setup is easier to interpret because all epochs are judged against the same validation subset.
- To compare stopping mechanisms, keep the same architecture and change only `stopping_rule`.

References
- Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford University Press.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
