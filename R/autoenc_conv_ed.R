#'@title Convolutional Autoencoder - Encode-Decode
#'@description Creates a deep learning convolutional autoencoder (ConvAE) to encode and decode
#' sequences of observations. Wraps a PyTorch implementation.
#'@param input_size Integer. Number of input features per observation.
#'@param encoding_size Integer. Size of the latent (bottleneck) representation.
#'@param batch_size Integer. Mini-batch size used during training. Default is 32.
#'@param epochs Integer. Maximum number of training epochs. Default is 100.
#'@param num_epochs Deprecated compatibility alias for `epochs`. If informed, it overrides `epochs`.
#'@param learning_rate Numeric. Optimizer learning rate. Default is 0.001.
#'@param validation_strategy Character. One of `static` or `dynamic`.
#'@param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#'@param val_ratio Numeric. Validation fraction used when validation is enabled. Default is 0.3.
#'@param patience Integer. Early stopping patience. Default is 100.
#'@param min_delta Numeric. Minimum improvement to reset early stopping. Default is 1e-4.
#'@param sma_window Integer. Window size used by `sma`. Default is 5.
#'@param ema_alpha Numeric. Smoothing factor used by `ema`. Default is 0.2.
#'@param test_window Integer. Window size used by `h`. Default is 30.
#'@param p_value Numeric. Significance threshold used by `h`. Default is 0.05.
#'@return A `autoenc_conv_ed` object.
#'@examples
#'\dontrun{
#'X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#'ae <- autoenc_conv_ed(input_size = 20, encoding_size = 5, epochs = 100)
#'ae <- daltoolbox::fit(ae, X)
#'X_hat <- daltoolbox::transform(ae, X)  # same dims as X
#'mean((X - X_hat)^2)
#'}
#'
#'# See:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_conv_ed.md
#'@importFrom daltoolbox autoenc_base_ed
#'@import reticulate
#'@export
autoenc_conv_ed <- function(input_size, encoding_size, batch_size = 32, epochs = 100L, num_epochs = NULL, learning_rate = 0.001,
                            validation_strategy = c("static", "dynamic"),
                            stopping_rule = c("none", "patience", "sma", "ema", "h"),
                            val_ratio = 0.3, patience = 100L, min_delta = 1e-4,
                            sma_window = 5L, ema_alpha = 0.2, test_window = 30L, p_value = 0.05) {
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- daltoolbox::autoenc_base_ed(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$epochs <- resolve_autoenc_epochs(epochs, num_epochs)
  obj$num_epochs <- obj$epochs
  obj$learning_rate <- learning_rate
  obj$validation_strategy <- validation_strategy
  obj$stopping_rule <- stopping_rule
  obj$val_ratio <- val_ratio
  obj$patience <- patience
  obj$min_delta <- min_delta
  obj$sma_window <- sma_window
  obj$ema_alpha <- ema_alpha
  obj$test_window <- test_window
  obj$p_value <- p_value
  class(obj) <- append("autoenc_conv_ed", class(obj))

  return(obj)
}

#'@export
fit.autoenc_conv_ed <- function(obj, data, ...) {
  if (!exists("autoenc_conv_create"))
    reticulate::source_python(system.file("python", "autoenc_conv.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- autoenc_conv_create(obj$input_size, obj$encoding_size, validation_strategy = obj$validation_strategy, stopping_rule = obj$stopping_rule)

  result <- autoenc_conv_fit(obj$model, data, batch_size = obj$batch_size, num_epochs = obj$epochs, learning_rate = obj$learning_rate,
                             validation_strategy = obj$validation_strategy, stopping_rule = obj$stopping_rule, val_ratio = obj$val_ratio,
                             patience = obj$patience, min_delta = obj$min_delta, sma_window = obj$sma_window, ema_alpha = obj$ema_alpha,
                             test_window = obj$test_window, p_value = obj$p_value)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@export
transform.autoenc_conv_ed <- function(obj, data, ...) {
  if (!exists("autoenc_conv_create"))
    reticulate::source_python(system.file("python", "autoenc_conv.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    # Reconstruct inputs using the trained convolutional autoencoder
    result <- autoenc_conv_encode_decode(obj$model, data)
  }

  return(result)
}