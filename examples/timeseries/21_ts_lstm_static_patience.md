## 21. LSTM Regression with Static Validation and Patience

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
##             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732 0.5984721
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732 0.5984721 0.3816610
```

Before moving on, we visualize the series so the effect of the next transformation can be compared against the original signal.


``` r
# Series visualization
plot_ts(x = tsd$x, y = tsd$y) + theme(text = element_text(size = 16))
```

![plot of chunk unnamed-chunk-4](fig/21_ts_lstm_static_patience/unnamed-chunk-4-1.png)

We now preserve the time order, split the data into train and test partitions, and project the windows into inputs and targets.


``` r
# Train-test split and projection (X, y)

samp <- ts_sample(ts, test_size = 5)
io_train <- ts_projection(samp$train)
io_test <- ts_projection(samp$test)
```

Contract note
- `fit()` consumes the supervised projection `x = io_train$input` and `y = io_train$output`.
- `predict()` is interpreted here as the numeric forecast path; `as.vector()` keeps that explicit for downstream code.

We now train the LSTM model with a fixed validation partition and patience-based early stopping.


``` r
# Training the LSTM model

model <- ts_lstm(
  ts_norm_gminmax(),
  input_size = 9,
  sequence_length = 3L,
  hidden_size = 16L,
  epochs = 300L,
  validation_strategy = "static",
  stopping_rule = "patience",
  patience = 20L
)
set_example_seed()
model <- fit(model, x = io_train$input, y = io_train$output)
```

Constructor configuration
- `validation_strategy = "static"` keeps the validation subset fixed across the whole run.
- `stopping_rule = "patience"` stops training when validation loss stops improving for the configured number of epochs.
- `epochs = 300L` is only an upper bound here; the effective training length is controlled by early stopping.
- The final curve plot shows both `train_loss_hist` and `val_loss_hist`.

Architecture variations
- `sequence_length` converts each row into a true multistep sequence instead of a single recurrent step.
- `hidden_size`, `num_layers`, `dropout`, and `bidirectional` change the recurrent backbone.
- `mlp_hidden_sizes` adds a dense head after the final recurrent state.
- In this example, `input_size = 9` matches the nine lagged predictors produced by `ts_projection()`, and `sequence_length = 3L` turns them into three temporal steps with three features each.
- This example emphasizes interpretability of the validation curve, because all epochs are judged against the same holdout partition.

We first evaluate the in-sample fit so the model adjustment can be compared with the later forecast.


``` r
# Fit evaluation (train)

adjust <- predict(model, io_train$input)
adjust <- as.vector(adjust)
output <- as.vector(io_train$output)
ev_adjust <- evaluate(model, output, adjust)
ev_adjust$mse
```

```
## [1] 0.0006032538
```

We now forecast the test set and compare the predicted values with the observed ones.


``` r
# Forecast on test set

steps_ahead <- 1
prediction <- predict(model, x = io_test$input, steps_ahead = steps_ahead)
prediction <- as.vector(prediction)

output <- as.vector(io_test$output)
if (steps_ahead > 1)
  output <- output[1:steps_ahead]

print(sprintf("%.2f, %.2f", output, prediction))
```

```
## [1] "0.41, 0.40"   "0.17, 0.18"   "-0.08, -0.05" "-0.32, -0.29" "-0.54, -0.53"
```

This chunk evaluates the custom component on the held-out test segment.


``` r
# Test evaluation

ev_test <- evaluate(model, output, prediction)
print(head(ev_test$metrics))
```

```
##            mse     smape        R2
## 1 0.0003237804 0.1067715 0.9972035
```

``` r
print(sprintf("smape: %.2f", 100 * ev_test$metrics$smape))
```

```
## [1] "smape: 10.68"
```

This final plot summarizes the result of the transformation so the effect can be interpreted visually.


``` r
# Plot results

yvalues <- c(io_train$output, io_test$output)
plot_ts_pred(y = yvalues, yadj = adjust, ypre = prediction) + theme(text = element_text(size = 16))
```

![plot of chunk unnamed-chunk-10](fig/21_ts_lstm_static_patience/unnamed-chunk-10-1.png)

The additional plot below shows the training curve and, when enabled, the validation curve used by the unified early-stopping strategies.


``` r
# Training and validation curves

fit_loss <- data.frame(
  x = seq_along(model$train_loss_hist),
  train_loss = model$train_loss_hist
)

if (!is.null(model$val_loss_hist) && length(model$val_loss_hist) > 0) {
  fit_loss$val_loss <- model$val_loss_hist
}

colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
grf <- plot_series(fit_loss, colors = colors)
plot(grf)
```

![plot of chunk unnamed-chunk-11](fig/21_ts_lstm_static_patience/unnamed-chunk-11-1.png)


``` r
# Effective training duration
print(model$epochs_done)
```

```
## [1] 165
```

Notes
- This configuration is useful when you want a stable validation reference across epochs.
- To compare with redrawn validation splits, see `22_ts_lstm_dynamic_patience`.

References
- S. Hochreiter and J. Schmidhuber (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.
