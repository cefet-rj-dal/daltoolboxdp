#' @title Variational Autoencoder - Encode
#' @description Creates a variational autoencoder (VAE) with configurable dense encoder/decoder
#'   blocks through a Python/PyTorch backend.
#'
#' @details
#' The VAE now exposes the hidden layout of both encoder and decoder, the activation family,
#' the latent reconstruction head, and the reconstruction loss. This makes it possible to
#' move from the original `64 -> 32 -> latent` structure to deeper or shallower alternatives.
#'
#' @param input_size Integer. Number of input features per observation.
#' @param encoding_size Integer. Size of the latent (bottleneck) representation.
#' @param encoder_hidden_sizes Integer vector used by the encoder backbone. Default is
#'   `c(64L, 32L)`, matching the previous implementation.
#' @param decoder_hidden_sizes Optional integer vector used by the decoder backbone.
#'   If `NULL`, the decoder mirrors `encoder_hidden_sizes` in reverse order.
#' @param activation Character. Hidden activation function. One of
#'   `"leaky_relu"`, `"relu"`, `"elu"`, `"gelu"`, or `"tanh"`.
#' @param negative_slope Numeric. Negative slope used when `activation = "leaky_relu"`.
#' @param output_activation Character. Output activation of the decoder. One of
#'   `"sigmoid"`, `"none"`, `"relu"`, `"tanh"`, or `"softplus"`.
#' @param reconstruction_loss Character. Reconstruction term used in the ELBO. One of
#'   `"bce"` or `"mse"`.
#' @param batch_size Integer. Mini-batch size used during training. Default is 32.
#' @param epochs Integer. Maximum number of training epochs. Default is 100.
#' @param num_epochs Deprecated compatibility alias for `epochs`. If informed, it overrides `epochs`.
#' @param learning_rate Numeric. Optimizer learning rate. Default is 0.001.
#' @param validation_strategy Character. One of `static` or `dynamic`.
#' @param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#' @param val_ratio Numeric. Validation fraction used when validation is enabled. Default is 0.3.
#' @param patience Integer. Early stopping patience. Default is 100.
#' @param min_delta Numeric. Minimum improvement to reset early stopping. Default is 1e-4.
#' @param sma_window Integer. Window size used by `sma`. Default is 5.
#' @param ema_alpha Numeric. Smoothing factor used by `ema`. Default is 0.2.
#' @param test_window Integer. Window size used by `h`. Default is 30.
#' @param p_value Numeric. Significance threshold used by `h`. Default is 0.05.
#' @return A `autoenc_variational_e` object.
#'
#' @references
#' Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_variational_e(
#'   input_size = 20,
#'   encoding_size = 5,
#'   encoder_hidden_sizes = c(128L, 64L, 32L),
#'   reconstruction_loss = "mse"
#' )
#' ae <- daltoolbox::fit(ae, X)
#' Z <- daltoolbox::transform(ae, X)
#' dim(Z)
#' }
#'
#' @importFrom daltoolbox autoenc_base_e
#' @import reticulate
#' @export
autoenc_variational_e <- function(input_size, encoding_size,
                                  encoder_hidden_sizes = c(64L, 32L),
                                  decoder_hidden_sizes = NULL,
                                  activation = c("leaky_relu", "relu", "elu", "gelu", "tanh"),
                                  negative_slope = 0.2,
                                  output_activation = c("sigmoid", "none", "relu", "tanh", "softplus"),
                                  reconstruction_loss = c("bce", "mse"),
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
  reconstruction_loss <- match.arg(reconstruction_loss)
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$encoder_hidden_sizes <- normalize_hidden_sizes(encoder_hidden_sizes)
  obj$decoder_hidden_sizes <- normalize_hidden_sizes(decoder_hidden_sizes)
  obj$activation <- activation
  obj$negative_slope <- negative_slope
  obj$output_activation <- output_activation
  obj$reconstruction_loss <- reconstruction_loss
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
  class(obj) <- append("autoenc_variational_e", class(obj))

  obj
}

#' @exportS3Method fit autoenc_variational_e
fit.autoenc_variational_e <- function(obj, data, ...) {
  if (!exists("autoenc_variational_create"))
    reticulate::source_python(system.file("python", "autoenc_variational.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- autoenc_variational_create(
      obj$input_size,
      obj$encoding_size,
      encoder_hidden_sizes = obj$encoder_hidden_sizes,
      decoder_hidden_sizes = obj$decoder_hidden_sizes,
      activation = obj$activation,
      negative_slope = obj$negative_slope,
      output_activation = obj$output_activation,
      reconstruction_loss = obj$reconstruction_loss,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  result <- autoenc_variational_fit(
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

#' @exportS3Method transform autoenc_variational_e
transform.autoenc_variational_e <- function(obj, data, ...) {
  if (!exists("autoenc_variational_create"))
    reticulate::source_python(system.file("python", "autoenc_variational.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_variational_encode(obj$model, data)
  }
  result
}