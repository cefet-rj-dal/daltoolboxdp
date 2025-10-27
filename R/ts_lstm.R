#'@title LSTM
#'@description Time series forecaster using an LSTM neural network.
#' Wraps a PyTorch implementation via `reticulate`.
#'
#'@param preprocess Optional preprocessing/normalization object.
#'@param input_size Integer. Number of lagged inputs per training example.
#'@param epochs Integer. Maximum number of training epochs.
#'@return A `ts_lstm` object.
#'
#'@references
#' Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
#'
#'@examples
#'\dontrun{
#'# Fit a simple LSTM forecaster (see vignette for end-to-end usage)
#'tsf <- ts_lstm(input_size = 12, epochs = 1000L)
#'}
#'
#'# See:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/timeseries/ts_lstm.md
#'@importFrom tspredit ts_regsw
#'@import reticulate
#'@export
ts_lstm <- function(preprocess = NA, input_size = NA, epochs = 10000L) {
  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$epochs <- epochs
  class(obj) <- append("ts_lstm", class(obj))

  return(obj)
}

#'@importFrom tspredit do_fit
#'@exportS3Method do_fit ts_lstm
do_fit.ts_lstm <- function(obj, x, y) {
  if (!exists("ts_lstm_create"))
    reticulate::source_python(system.file("python", "ts_lstm.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- ts_lstm_create(obj$input_size, obj$input_size)

  # Build training frame with target in column t0 as expected by Python code
  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- ts_lstm_fit(obj$model, df_train, obj$epochs, 0.001)

  return(obj)
}


#'@importFrom tspredit do_predict
#'@exportS3Method do_predict ts_lstm
do_predict.ts_lstm <- function(obj, x) {
  if (!exists("ts_lstm_predict"))
    reticulate::source_python(system.file("python", "ts_lstm.py", package = "daltoolboxdp"))

  # Prediction frame with dummy target column as required by Python code
  X_values <- as.data.frame(x)
  X_values$t0 <- 0

  n <- nrow(X_values)
  prediction <- ts_lstm_predict(obj$model, X_values)
  return(prediction)
}
