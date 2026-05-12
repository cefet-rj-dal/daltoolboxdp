#' @title Denoising Autoencoder - Encode
#' @description Creates a denoising autoencoder that learns robust latent representations
#'   from corrupted inputs through a Python/PyTorch backend.
#'
#' @details
#' Besides the denoising factor, this constructor exposes the same encoder/decoder
#' customization available in [autoenc_e()]. This allows the user to combine
#' shallow or deep dense architectures with stochastic input corruption.
#'
#' @param input_size Integer. Number of input features per observation.
#' @param encoding_size Integer. Size of the latent (bottleneck) representation.
#' @param encoder_hidden_sizes Integer vector. Hidden sizes used by the encoder.
#' @param decoder_hidden_sizes Optional integer vector. Hidden sizes used by the decoder.
#'   If `NULL`, the decoder mirrors `encoder_hidden_sizes` in reverse order.
#' @param activation Character. Hidden activation function. One of
#'   `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, `"selu"`, or `"tanh"`.
#' @param output_activation Character. Output activation of the decoder. One of
#'   `"none"`, `"relu"`, `"sigmoid"`, `"tanh"`, or `"softplus"`.
#' @param negative_slope Numeric. Negative slope used when `activation = "leaky_relu"`.
#' @param batch_size Integer. Mini-batch size used during training. Default is 32.
#' @param epochs Integer. Maximum number of training epochs. Default is 100.
#' @param num_epochs Deprecated compatibility alias for `epochs`. If informed, it overrides `epochs`.
#' @param learning_rate Numeric. Optimizer learning rate. Default is 0.001.
#' @param noise_factor Numeric. Standard deviation (scale) of the noise added during training.
#' @param validation_strategy Character. One of `static` or `dynamic`.
#' @param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#' @param val_ratio Numeric. Validation fraction used when validation is enabled. Default is 0.3.
#' @param patience Integer. Early stopping patience. Default is 100.
#' @param min_delta Numeric. Minimum improvement to reset early stopping. Default is 1e-4.
#' @param sma_window Integer. Window size used by `sma`. Default is 5.
#' @param ema_alpha Numeric. Smoothing factor used by `ema`. Default is 0.2.
#' @param test_window Integer. Window size used by `h`. Default is 30.
#' @param p_value Numeric. Significance threshold used by `h`. Default is 0.05.
#' @return A `autoenc_denoise_e` object.
#'
#' @references
#' Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and Composing Robust Features with Denoising Autoencoders.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_denoise_e(
#'   input_size = 20,
#'   encoding_size = 5,
#'   encoder_hidden_sizes = c(128L, 64L),
#'   noise_factor = 0.2
#' )
#' ae <- daltoolbox::fit(ae, X)
#' Z <- daltoolbox::transform(ae, X)
#' dim(Z)
#' }
#'
#' @importFrom daltoolbox autoenc_base_e
#' @import reticulate
#' @export
autoenc_denoise_e <- function(input_size, encoding_size,
                              encoder_hidden_sizes = 64L,
                              decoder_hidden_sizes = NULL,
                              activation = c("relu", "leaky_relu", "elu", "gelu", "selu", "tanh"),
                              output_activation = c("none", "relu", "sigmoid", "tanh", "softplus"),
                              negative_slope = 0.2,
                              batch_size = 32,
                              epochs = 100L,
                              num_epochs = NULL,
                              learning_rate = 0.001,
                              noise_factor = 0.3,
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
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
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
  obj$noise_factor <- noise_factor
  obj$validation_strategy <- validation_strategy
  obj$stopping_rule <- stopping_rule
  obj$val_ratio <- val_ratio
  obj$patience <- patience
  obj$min_delta <- min_delta
  obj$sma_window <- sma_window
  obj$ema_alpha <- ema_alpha
  obj$test_window <- test_window
  obj$p_value <- p_value
  class(obj) <- append("autoenc_denoise_e", class(obj))

  obj
}

#' @exportS3Method fit autoenc_denoise_e
fit.autoenc_denoise_e <- function(obj, data, ...) {
  if (!exists("autoenc_denoise_create"))
    reticulate::source_python(system.file("python", "autoenc_denoise.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- autoenc_denoise_create(
      obj$input_size,
      obj$encoding_size,
      noise_factor = obj$noise_factor,
      encoder_hidden_sizes = obj$encoder_hidden_sizes,
      decoder_hidden_sizes = obj$decoder_hidden_sizes,
      activation = obj$activation,
      output_activation = obj$output_activation,
      negative_slope = obj$negative_slope,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  result <- autoenc_denoise_fit(
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

#' @exportS3Method transform autoenc_denoise_e
transform.autoenc_denoise_e <- function(obj, data, ...) {
  if (!exists("autoenc_denoise_create"))
    reticulate::source_python(system.file("python", "autoenc_denoise.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_denoise_encode(obj$model, data)
  }
  result
}