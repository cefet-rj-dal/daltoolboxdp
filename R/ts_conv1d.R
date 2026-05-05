#'@title Conv1D
#'@description Time series forecaster using a 1D convolutional neural network
#' with unified training strategies.
#' Wraps a PyTorch implementation via `reticulate`.
#'
#'@param preprocess Optional preprocessing/normalization object.
#'@param input_size Integer. Number of lagged inputs per training example.
#'@param epochs Integer. Maximum number of training epochs. Default is `100L`.
#'@param lr Numeric. Optimizer learning rate.
#'@param validation_strategy Character. One of `static` or `dynamic`.
#'@param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#'@param val_ratio Numeric. Validation fraction used when validation is enabled.
#'@param batch_size Integer. Mini-batch size.
#'@param patience Integer. Early stopping patience.
#'@param min_delta Numeric. Minimum improvement to reset early stopping.
#'@param sma_window Integer. Window size used by `sma`.
#'@param ema_alpha Numeric. Smoothing factor used by `ema`.
#'@param test_window Integer. Window size used by `h`.
#'@param p_value Numeric. Significance threshold used by `h`.
#'@param seed Integer. Seed used by data splitting routines.
#'@return A `ts_conv1d` object.
#'@examples
#'\dontrun{
#'library(daltoolboxdp)
#'model <- ts_conv1d(input_size = 12, epochs = 100L)
#'}
#'@importFrom tspredit ts_regsw
#'@import reticulate
#'@export
ts_conv1d <- function(preprocess = NA,
                      input_size = NA,
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
                      p_value = 0.05,
                      seed = 42L) {
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)

  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$channels <- 1L
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
  obj$seed <- if (is.null(seed)) NULL else as.integer(seed)
  class(obj) <- append("ts_conv1d", class(obj))
  return(obj)
}

#'@importFrom tspredit do_fit
#'@exportS3Method do_fit ts_conv1d
do_fit.ts_conv1d <- function(obj, x, y) {
  if (!exists("ts_conv1d_create"))
    reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- ts_conv1d_create(
      obj$channels,
      obj$input_size,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  df_train <- as.data.frame(x)
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
    p_value = obj$p_value,
    seed = obj$seed
  )

  obj$train_loss_hist <- obj$model$train_loss_hist
  obj$val_loss_hist <- obj$model$val_loss_hist
  obj$epochs_done <- obj$model$epochs_done
  return(obj)
}

#'@importFrom tspredit do_predict
#'@exportS3Method do_predict ts_conv1d
do_predict.ts_conv1d <- function(obj, x) {
  if (!exists("ts_conv1d_predict"))
    reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolboxdp"))

  x_values <- as.data.frame(x)
  x_values$t0 <- 0
  ts_conv1d_predict(obj$model, x_values, batch_size = obj$batch_size)
}
