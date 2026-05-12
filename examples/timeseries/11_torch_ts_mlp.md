## Time-Series MLP Regression

About the method
- `torch_ts_mlp` uses a feedforward multilayer perceptron over lagged windows.
- It is the direct PyTorch MLP counterpart to the recurrent and convolutional forecasters already available in this package.

Didactic goal: compare a plain feedforward forecaster with `ts_lstm` and `ts_conv1d`, while preserving the same Experiment Line used in `tspredit`.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Time Series Regression - PyTorch MLP

# Installing packages (if needed)
#install.packages("daltoolboxdp")
```

We start by loading the packages used throughout this example.


``` r
# Loading the packages
library(daltoolbox)
library(daltoolboxdp)
library(tspredit)
library(ggplot2)
```

We load the example series that will be used throughout the demonstration.


``` r
# Series for study and sliding windows

data(tsd)
ts <- ts_data(tsd$y, 10)
ts_head(ts, 3)
```

```
##             t9        t8        t7        t6        t5        t4        t3
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974
##             t2        t1        t0
## [1,] 0.9839859 0.9092974 0.7780732
## [2,] 0.9092974 0.7780732 0.5984721
## [3,] 0.7780732 0.5984721 0.3816610
```

Before moving on, we visualize the series so the effect of the next transformation can be compared against the original signal.


``` r
# Series visualization
plot_ts(x = tsd$x, y = tsd$y) + theme(text = element_text(size = 16))
```

![plot of chunk unnamed-chunk-4](fig/11_torch_ts_mlp/unnamed-chunk-4-1.png)

We now preserve the time order, split the data into train and test partitions, and project the windows into inputs and targets.


``` r
# Train-test split and projection (X, y)

samp <- ts_sample(ts, test_size = 5)
io_train <- ts_projection(samp$train)
io_test <- ts_projection(samp$test)
```

We now train the MLP forecaster on the prepared training data.


``` r
# Training the PyTorch MLP model

model <- torch_ts_mlp(
  ts_norm_gminmax(),
  input_size = 4,
  hidden_sizes = c(64L, 32L, 16L),
  epochs = 1000L,
  batch_size = 16L
)
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
model <- fit(model, x = io_train$input, y = io_train$output)
```

```
## Error:
## ! object 'model' not found
```

Constructor configuration
- Fixed-epoch baseline: omit `epochs` to use the default value, keep `validation_strategy = "static"`, and `stopping_rule = "none"`.
- Static early stopping: keep `validation_strategy = "static"` and choose `stopping_rule = "patience"`, `"sma"`, `"ema"`, or `"h"`.
- Dynamic early stopping: switch `validation_strategy = "dynamic"` and reuse the same stopping rules.
- The final curve plot always shows `train_loss_hist`; it adds `val_loss_hist` when validation is active.

Architecture variations
- `activation` changes the hidden nonlinearity.
- `normalization = "batch"` or `"layer"` adds normalization after each hidden layer.
- `init_method` controls how linear weights are initialized.
- `output_activation` can constrain the forecast range when needed.

We first evaluate the in-sample fit so the model adjustment can be compared with the later forecast.


``` r
# Fit evaluation (train)

adjust <- predict(model, io_train$input)
```

```
## Error:
## ! object 'model' not found
```

``` r
adjust <- as.vector(adjust)
```

```
## Error:
## ! object 'adjust' not found
```

``` r
output <- as.vector(io_train$output)
ev_adjust <- evaluate(model, output, adjust)
```

```
## Error:
## ! object 'model' not found
```

``` r
ev_adjust$mse
```

```
## Error:
## ! object 'ev_adjust' not found
```

We now forecast the test set and compare the predicted values with the observed ones.


``` r
# Forecast on test set

steps_ahead <- 1
prediction <- predict(model, x = io_test$input, steps_ahead = steps_ahead)
```

```
## Error:
## ! object 'model' not found
```

``` r
prediction <- as.vector(prediction)
```

```
## Error:
## ! object 'prediction' not found
```

``` r
output <- as.vector(io_test$output)
if (steps_ahead > 1)
  output <- output[1:steps_ahead]

print(sprintf("%.2f, %.2f", output, prediction))
```

```
## Error:
## ! object 'prediction' not found
```

This chunk evaluates the custom component on the held-out test segment.


``` r
# Test evaluation

ev_test <- evaluate(model, output, prediction)
```

```
## Error:
## ! object 'model' not found
```

``` r
print(head(ev_test$metrics))
```

```
## Error:
## ! object 'ev_test' not found
```

``` r
print(sprintf("smape: %.2f", 100 * ev_test$metrics$smape))
```

```
## Error:
## ! object 'ev_test' not found
```

This final plot summarizes the result of the transformation so the effect can be interpreted visually.


``` r
# Plot results

yvalues <- c(io_train$output, io_test$output)
plot_ts_pred(y = yvalues, yadj = adjust, ypre = prediction) + theme(text = element_text(size = 16))
```

```
## Error:
## ! object 'adjust' not found
```

The additional plot below shows the training curve and, when enabled, the validation curve used by the unified early-stopping strategies.


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
- The default configuration is `validation_strategy = "static"` and `stopping_rule = "none"`, so only the training curve is shown.
- This example uses `epochs = 1000L` explicitly so the feedforward MLP reaches a competitive fit on this series.
- To display validation loss as well, use an early-stopping rule such as `"patience"`, `"sma"`, `"ema"`, or `"h"`.
- To combine dynamic validation with those criteria, set `validation_strategy = "dynamic"`.

References
- Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
