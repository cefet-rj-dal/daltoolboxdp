#'@title Conv1D
#'@description Time series forecaster using a 1D convolutional neural network.
#' Wraps a PyTorch implementation via `reticulate`.
#'
#'@param preprocess Optional preprocessing/normalization object.
#'@param input_size Integer. Number of lagged inputs per training example.
#'@param epochs Integer. Maximum number of training epochs.
#'@return A `ts_conv1d` object.
#'
#'@references
#' LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition.
#' Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.
#'
#'@examples
#'\dontrun{
#'# Fit a simple Conv1D forecaster (see linked vignette for full workflow)
#'tsf <- ts_conv1d(input_size = 12, epochs = 1000L)
#'}
#'
#'# See:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/timeseries/ts_conv1d.md
#'@importFrom tspredit ts_regsw
#'@import reticulate
#'@export
ts_conv1d <- function(preprocess = NA, input_size = NA, epochs = 10000L) {
  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$channels <- 1
  obj$epochs <- epochs
  class(obj) <- append("ts_conv1d", class(obj))
  return(obj)
}

#'@importFrom tspredit do_fit
#'@exportS3Method do_fit ts_conv1d
do_fit.ts_conv1d <- function(obj, x, y) {
  # Load backing Python implementation
  reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- ts_conv1d_create(obj$channels, obj$input_size)

  # Build training frame with target in column t0 as expected by Python code
  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- ts_conv1d_fit(obj$model, df_train, obj$epochs, 0.001)

  return(obj)
}

#'@importFrom tspredit do_predict
#'@exportS3Method do_predict ts_conv1d
do_predict.ts_conv1d <- function(obj, x) {
  # Load backing Python implementation if needed
  reticulate::source_python(system.file("python", "ts_conv1d.py", package = "daltoolboxdp"))

  # Prediction frame with dummy target column as required by Python code
  X_values <- as.data.frame(x)
  X_values$t0 <- 0

  n <- nrow(X_values)
  prediction <- ts_conv1d_predict(obj$model, X_values)
  return(prediction)
}
