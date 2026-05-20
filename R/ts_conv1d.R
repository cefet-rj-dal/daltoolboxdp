#' @title Conv1D
#' @description Time-series forecaster using a configurable 1D convolutional neural network
#'   with unified training strategies and a Python/PyTorch backend.
#'
#' @details
#' The Conv1D forecaster now supports multiple convolutional blocks, explicit channel/sequence
#' reshaping, optional pooling, and a configurable dense prediction head. Keep the defaults to
#' preserve the original single-channel behavior, or define architectures such as
#' `conv_channels = c(32L, 64L)` and `dense_hidden_sizes = c(64L, 16L)`.
#'
#' The object follows the `tspredit::ts_regsw()` contract: `fit()` receives
#' supervised lag matrices and `predict()` returns a plain numeric vector, even
#' when upstream time-series wrappers attach auxiliary metadata to forecast
#' objects.
#'
#' @param preprocess Optional preprocessing/normalization object.
#' @param input_size Integer. Number of lagged inputs per training example.
#' @param input_map Lag-selection strategy object, typically created by
#'   `tspredit::ts_lagmap()`.
#' @param in_channels Integer. Number of channels used to reshape each example before the convolution.
#'   `input_size` must equal `in_channels * sequence_length`.
#' @param sequence_length Optional integer. Temporal length after reshaping. If `NULL`, it is inferred
#'   as `input_size / in_channels`.
#' @param conv_channels Integer vector. Output channels for each convolutional block.
#' @param kernel_sizes Integer vector. Kernel sizes for each convolutional block. If `NULL`, defaults to
#'   `2L` for sequence lengths greater than 1 and `1L` otherwise.
#' @param strides Integer vector. Strides for each convolutional block.
#' @param pooling Character. Pooling strategy applied after each convolutional block. One of
#'   `"none"`, `"max"`, or `"avg"`.
#' @param pool_kernel_size Integer. Pooling kernel size when pooling is enabled.
#' @param dense_hidden_sizes Integer vector. Hidden sizes of the dense head after the convolutional stack.
#' @param activation Character. Activation function used in convolutional and dense hidden layers. One of
#'   `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, or `"tanh"`.
#' @param epochs Integer. Maximum number of training epochs. Default is `100L`.
#' @param lr Numeric. Optimizer learning rate.
#' @param validation_strategy Character. One of `static` or `dynamic`.
#' @param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#' @param val_ratio Numeric. Validation fraction used when validation is enabled.
#' @param batch_size Integer. Mini-batch size.
#' @param patience Integer. Early stopping patience.
#' @param min_delta Numeric. Minimum improvement to reset early stopping.
#' @param sma_window Integer. Window size used by `sma`.
#' @param ema_alpha Numeric. Smoothing factor used by `ema`.
#' @param test_window Integer. Window size used by `h`.
#' @param p_value Numeric. Significance threshold used by `h`.
#' @return A `ts_conv1d` object.
#' @examples
#' \dontrun{
#' library(daltoolboxdp)
#' model <- ts_conv1d(
#'   input_size = 12,
#'   input_map = tspredit::ts_lagmap("acf"),
#'   in_channels = 1L,
#'   conv_channels = c(32L, 64L),
#'   dense_hidden_sizes = c(64L, 16L),
#'   epochs = 100L
#' )
#' }
#' @importFrom tspredit ts_regsw
#' @import reticulate
#' @export
ts_conv1d <- function(preprocess = NA,
                      input_size = NA,
                      input_map = tspredit::ts_lagmap(),
                      in_channels = 1L,
                      sequence_length = NULL,
                      conv_channels = 64L,
                      kernel_sizes = NULL,
                      strides = 1L,
                      pooling = c("none", "max", "avg"),
                      pool_kernel_size = 2L,
                      dense_hidden_sizes = 50L,
                      activation = c("relu", "leaky_relu", "elu", "gelu", "tanh"),
                      epochs = 100L,
                      lr = 0.001,
                      validation_strategy = c("static", "dynamic"),
                      stopping_rule = c("none", "patience", "sma", "ema", "h"),
                      val_ratio = 0.2,
                      batch_size = 8L,
                      patience = 100L,
                      min_delta = 1e-4,
                      sma_window = 5L,
                      ema_alpha = 0.2,
                      test_window = 30L,
                      p_value = 0.05) {
  pooling <- match.arg(pooling)
  activation <- match.arg(activation)
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)

  obj <- tspredit::ts_regsw(preprocess, input_size, input_map)
  obj$in_channels <- as.integer(in_channels)
  obj$sequence_length <- if (is.null(sequence_length)) NULL else as.integer(sequence_length)
  obj$conv_channels <- as.integer(conv_channels)
  obj$kernel_sizes <- if (is.null(kernel_sizes)) NULL else as.integer(kernel_sizes)
  obj$strides <- as.integer(strides)
  obj$pooling <- pooling
  obj$pool_kernel_size <- as.integer(pool_kernel_size)
  obj$dense_hidden_sizes <- as.integer(dense_hidden_sizes)
  obj$activation <- activation
  obj$epochs <- as.integer(epochs)
  obj$lr <- as.numeric(lr)
  obj$validation_strategy <- validation_strategy
  obj$stopping_rule <- stopping_rule
  obj$val_ratio <- as.numeric(val_ratio)
  obj$batch_size <- as.integer(batch_size)
  obj$patience <- as.integer(patience)
  obj$min_delta <- as.numeric(min_delta)
  obj$sma_window <- as.integer(sma_window)
  obj$ema_alpha <- as.numeric(ema_alpha)
  obj$test_window <- as.integer(test_window)
  obj$p_value <- as.numeric(p_value)
  class(obj) <- append("ts_conv1d", class(obj))
  obj
}

#' @importFrom tspredit do_fit
#' @exportS3Method do_fit ts_conv1d
do_fit.ts_conv1d <- function(obj, x, y) {
  if (!exists("ts_conv1d_create"))
    reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- ts_conv1d_create(
      obj$in_channels,
      obj$input_size,
      sequence_length = obj$sequence_length,
      conv_channels = obj$conv_channels,
      kernel_sizes = obj$kernel_sizes,
      strides = obj$strides,
      pooling = obj$pooling,
      pool_kernel_size = obj$pool_kernel_size,
      dense_hidden_sizes = obj$dense_hidden_sizes,
      activation = obj$activation,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  df_train <- as.data.frame(x)
  # Keep the target column as a plain vector for the Python backend.
  df_train$t0 <- as.vector(y)

  obj$model <- ts_conv1d_fit(
    obj$model,
    df_train,
    n_epochs = obj$epochs,
    lr = obj$lr,
    validation_strategy = obj$validation_strategy,
    stopping_rule = obj$stopping_rule,
    val_ratio = obj$val_ratio,
    batch_size = obj$batch_size,
    patience = obj$patience,
    min_delta = obj$min_delta,
    sma_window = obj$sma_window,
    ema_alpha = obj$ema_alpha,
    test_window = obj$test_window,
    p_value = obj$p_value
  )

  obj$train_loss_hist <- obj$model$train_loss_hist
  obj$val_loss_hist <- obj$model$val_loss_hist
  obj$epochs_done <- obj$model$epochs_done
  obj
}

#' @importFrom tspredit do_predict
#' @exportS3Method do_predict ts_conv1d
do_predict.ts_conv1d <- function(obj, x) {
  if (!exists("ts_conv1d_predict"))
    reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolboxdp"))

  x_values <- as.data.frame(x)
  x_values$t0 <- 0
  # Return only the numeric forecast path expected by tspredit and downstream
  # wrappers such as harbinger.
  as.vector(ts_conv1d_predict(obj$model, x_values, batch_size = obj$batch_size))
}
