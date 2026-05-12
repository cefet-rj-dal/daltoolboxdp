## LSTM Regression

About the method
- LSTM networks are recurrent models designed to retain and update information across ordered inputs.
- They are useful when forecasting depends on sequential dependencies that may span more than a few immediate lags.

Didactic goal: compare a recurrent sequence model with the feedforward and convolutional alternatives, while preserving the same Experiment Line used in `tspredit`.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Time Series Regression - LSTM

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

![plot of chunk unnamed-chunk-4](fig/13_ts_lstm/unnamed-chunk-4-1.png)

We now preserve the time order, split the data into train and test partitions, and project the windows into inputs and targets.


``` r
# Train-test split and projection (X, y)

samp <- ts_sample(ts, test_size = 5)
io_train <- ts_projection(samp$train)
io_test <- ts_projection(samp$test)
```

We now train the LSTM model on the prepared training data.


``` r
# Training the LSTM model

model <- ts_lstm(
  ts_norm_gminmax(),
  input_size = 9,
  sequence_length = 3L,
  hidden_size = 16L,
  epochs = 200L
)
```

```
## Error:
## ! read failed on C:/R/R-4.5.0/library/daltoolboxdp/R/daltoolboxdp.rdb
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
- Fixed-epoch baseline: omit `epochs` to use the default value of `100L`, keep `validation_strategy = "static"`, and `stopping_rule = "none"`.
- Static early stopping: keep `validation_strategy = "static"` and choose `stopping_rule = "patience"`, `"sma"`, `"ema"`, or `"h"`.
- Dynamic early stopping: switch `validation_strategy = "dynamic"` and reuse the same stopping rules.
- The final curve plot always shows `train_loss_hist`; it adds `val_loss_hist` when validation is active.

Architecture variations
- `sequence_length` converts each row into a true multistep sequence instead of a single recurrent step.
- `hidden_size`, `num_layers`, `dropout`, and `bidirectional` change the recurrent backbone.
- `mlp_hidden_sizes` adds a dense head after the final recurrent state.
- In this example, `input_size = 9` matches the nine lagged predictors produced by `ts_projection()`, and `sequence_length = 3L` turns them into three temporal steps with three features each.
- The example also raises `epochs` to `200L` on purpose, because the default `100L` is intended to be tuned by the user when needed.

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
- To display validation loss as well, use an early-stopping rule such as `"patience"`, `"sma"`, `"ema"`, or `"h"`.
- To combine dynamic validation with those criteria, set `validation_strategy = "dynamic"`.

References
- S. Hochreiter and J. Schmidhuber (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.
