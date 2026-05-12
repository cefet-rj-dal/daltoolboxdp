#' @title Autoencoder - Encode-Decode
#' @description Creates a dense autoencoder that compresses and reconstructs observations
#'   through a Python/PyTorch backend.
#'
#' @details
#' This variant exposes the same architecture controls as [autoenc_e()], but the
#' transformation returns reconstructions in the original input space. Use it when
#' reconstruction quality is part of the analysis or when the autoencoder is used
#' for anomaly detection.
#'
#' @inheritParams autoenc_e
#'
#' @return A `autoenc_ed` object.
#'
#' @references
#' Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks.
#' Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_ed(
#'   input_size = 20,
#'   encoding_size = 5,
#'   encoder_hidden_sizes = c(128L, 64L),
#'   decoder_hidden_sizes = c(64L, 128L)
#' )
#' ae <- daltoolbox::fit(ae, X)
#' X_hat <- daltoolbox::transform(ae, X)
#' mean((X - X_hat)^2)
#' }
#'
#' @importFrom daltoolbox autoenc_base_ed
#' @import reticulate
#' @export
autoenc_ed <- function(input_size, encoding_size,
                       encoder_hidden_sizes = 64L,
                       decoder_hidden_sizes = NULL,
                       activation = c("relu", "leaky_relu", "elu", "gelu", "selu", "tanh"),
                       output_activation = c("none", "relu", "sigmoid", "tanh", "softplus"),
                       negative_slope = 0.2,
                       batch_size = 32,
                       epochs = 100L,
                       num_epochs = NULL,
                       learning_rate = 0.001,
                       validation_strategy = c("static", "dynamic"),
                       stopping_rule = c("none", "patience", "sma", "ema", "h"),
                       val_ratio = 0.3,
                       patience = 100L,
                       min_delta = 1e-4,
                       sma_window = 5L,
                       ema_alpha = 0.2,
                       test_window = 30L,
                       p_value = 0.05) {
  activation <- match.arg(activation)
  output_activation <- match.arg(output_activation)
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- daltoolbox::autoenc_base_ed(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$encoder_hidden_sizes <- normalize_hidden_sizes(encoder_hidden_sizes)
  obj$decoder_hidden_sizes <- normalize_hidden_sizes(decoder_hidden_sizes)
  obj$activation <- activation
  obj$output_activation <- output_activation
  obj$negative_slope <- negative_slope
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
  class(obj) <- append("autoenc_ed", class(obj))

  obj
}

#' @exportS3Method fit autoenc_ed
fit.autoenc_ed <- function(obj, data, ...) {
  if (!exists("autoenc_create"))
    reticulate::source_python(system.file("python", "autoenc.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- autoenc_create(
      obj$input_size,
      obj$encoding_size,
      encoder_hidden_sizes = obj$encoder_hidden_sizes,
      decoder_hidden_sizes = obj$decoder_hidden_sizes,
      activation = obj$activation,
      output_activation = obj$output_activation,
      negative_slope = obj$negative_slope,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  result <- autoenc_fit(
    obj$model,
    data,
    batch_size = obj$batch_size,
    num_epochs = obj$epochs,
    learning_rate = obj$learning_rate,
    validation_strategy = obj$validation_strategy,
    stopping_rule = obj$stopping_rule,
    val_ratio = obj$val_ratio,
    patience = obj$patience,
    min_delta = obj$min_delta,
    sma_window = obj$sma_window,
    ema_alpha = obj$ema_alpha,
    test_window = obj$test_window,
    p_value = obj$p_value
  )

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  obj
}

#' @exportS3Method transform autoenc_ed
transform.autoenc_ed <- function(obj, data, ...) {
  if (!exists("autoenc_create"))
    reticulate::source_python(system.file("python", "autoenc.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_encode_decode(obj$model, data, batch_size = obj$batch_size)
  }
  result
}