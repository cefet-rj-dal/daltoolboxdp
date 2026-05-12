#' @title LSTM
#' @description Time-series forecaster using a configurable LSTM neural network with unified
#'   training strategies and a Python/PyTorch backend.
#'
#' @details
#' The LSTM forecaster now supports multiple recurrent layers, dropout, bidirectionality,
#' an optional dense head after the recurrent block, and explicit reshaping of each row into
#' a sequence via `sequence_length`. Keeping `sequence_length = 1L` reproduces the previous behavior.
#'
#' @param preprocess Optional preprocessing/normalization object.
#' @param input_size Integer. Number of lagged inputs per training example.
#' @param hidden_size Optional integer. Hidden size used inside the LSTM. If `NULL`, defaults to `input_size`.
#' @param sequence_length Integer. Number of time steps represented by each row. `input_size`
#'   must be divisible by `sequence_length`. Default is `1L`.
#' @param num_layers Integer. Number of LSTM layers.
#' @param dropout Numeric. Recurrent dropout applied between LSTM layers when `num_layers > 1`.
#' @param bidirectional Logical. Whether the LSTM is bidirectional.
#' @param mlp_hidden_sizes Integer vector. Hidden sizes of the dense head applied after the LSTM output.
#' @param activation Character. Activation function used in the dense head. One of
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
#' @return A `ts_lstm` object.
#' @examples
#' \dontrun{
#' library(daltoolboxdp)
#' model <- ts_lstm(
#'   input_size = 12,
#'   hidden_size = 16L,
#'   sequence_length = 3L,
#'   num_layers = 2L,
#'   dropout = 0.1,
#'   mlp_hidden_sizes = c(16L, 8L),
#'   epochs = 100L
#' )
#' }
#' @importFrom tspredit ts_regsw
#' @import reticulate
#' @export
ts_lstm <- function(preprocess = NA,
                    input_size = NA,
                    hidden_size = NULL,
                    sequence_length = 1L,
                    num_layers = 1L,
                    dropout = 0,
                    bidirectional = FALSE,
                    mlp_hidden_sizes = integer(0),
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
  activation <- match.arg(activation)
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)

  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$hidden_size <- if (is.null(hidden_size)) as.integer(input_size) else as.integer(hidden_size)
  obj$sequence_length <- as.integer(sequence_length)
  obj$num_layers <- as.integer(num_layers)
  obj$dropout <- as.numeric(dropout)
  obj$bidirectional <- as.logical(bidirectional)
  obj$mlp_hidden_sizes <- as.integer(mlp_hidden_sizes)
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
  class(obj) <- append("ts_lstm", class(obj))

  obj
}

#' @importFrom tspredit do_fit
#' @exportS3Method do_fit ts_lstm
do_fit.ts_lstm <- function(obj, x, y) {
  if (!exists("ts_lstm_create"))
    reticulate::source_python(system.file("python", "ts_lstm.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- ts_lstm_create(
      obj$hidden_size,
      obj$input_size,
      sequence_length = obj$sequence_length,
      num_layers = obj$num_layers,
      dropout = obj$dropout,
      bidirectional = obj$bidirectional,
      mlp_hidden_sizes = obj$mlp_hidden_sizes,
      activation = obj$activation,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- ts_lstm_fit(
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
#' @exportS3Method do_predict ts_lstm
do_predict.ts_lstm <- function(obj, x) {
  if (!exists("ts_lstm_predict"))
    reticulate::source_python(system.file("python", "ts_lstm.py", package = "daltoolboxdp"))

  x_values <- as.data.frame(x)
  x_values$t0 <- 0
  ts_lstm_predict(obj$model, x_values, batch_size = obj$batch_size)
}