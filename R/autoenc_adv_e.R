#' @title Adversarial Autoencoder - Encode
#' @description Creates an adversarial autoencoder (AAE) with configurable encoder,
#'   decoder and discriminator topologies through a Python/PyTorch backend.
#'
#' @details
#' The adversarial autoencoder exposes the latent prior scale, dropout, activation family,
#' and optimizer learning rates for each adversarial component. If the component-specific
#' learning rates are left as `NULL`, the wrapper derives them from `learning_rate`
#' using the training defaults of the Python implementation.
#'
#' @param input_size Integer. Number of input features per observation.
#' @param encoding_size Integer. Size of the latent (bottleneck) representation.
#' @param encoder_hidden_sizes Integer vector used by the encoder network. Default is `c(60L, 60L)`.
#' @param decoder_hidden_sizes Integer vector used by the decoder network. Default is `c(60L, 60L)`.
#' @param discriminator_hidden_sizes Integer vector used by the discriminator network.
#'   Default is `c(60L, 60L)`.
#' @param activation Character. Hidden activation function. One of
#'   `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, or `"tanh"`.
#' @param dropout Numeric. Dropout rate applied to adversarial hidden layers.
#' @param latent_prior_scale Numeric. Standard deviation scale used to sample the latent prior.
#' @param lr_encoder Optional numeric. Learning rate of the encoder reconstruction optimizer.
#' @param lr_decoder Optional numeric. Learning rate of the decoder reconstruction optimizer.
#' @param lr_generator Optional numeric. Learning rate of the encoder adversarial optimizer.
#' @param lr_discriminator Optional numeric. Learning rate of the discriminator optimizer.
#' @param batch_size Integer. Mini-batch size used during training. Default is 350.
#' @param epochs Integer. Maximum number of training epochs. Default is 100.
#' @param num_epochs Deprecated compatibility alias for `epochs`. If informed, it overrides `epochs`.
#' @param learning_rate Numeric. Base optimizer learning rate. Default is 0.001.
#' @param validation_strategy Character. One of `static` or `dynamic`.
#' @param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#' @param val_ratio Numeric. Validation fraction used when validation is enabled. Default is 0.3.
#' @param patience Integer. Early stopping patience. Default is 100.
#' @param min_delta Numeric. Minimum improvement to reset early stopping. Default is 1e-4.
#' @param sma_window Integer. Window size used by `sma`. Default is 5.
#' @param ema_alpha Numeric. Smoothing factor used by `ema`. Default is 0.2.
#' @param test_window Integer. Window size used by `h`. Default is 30.
#' @param p_value Numeric. Significance threshold used by `h`. Default is 0.05.
#' @return A `autoenc_adv_e` object.
#'
#' @references
#' Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2016). Adversarial Autoencoders.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_adv_e(
#'   input_size = 20,
#'   encoding_size = 5,
#'   encoder_hidden_sizes = c(128L, 64L),
#'   discriminator_hidden_sizes = c(64L, 32L),
#'   latent_prior_scale = 2
#' )
#' ae <- daltoolbox::fit(ae, X)
#' Z <- daltoolbox::transform(ae, X)
#' }
#'
#' @importFrom daltoolbox autoenc_base_e
#' @import reticulate
#' @export
autoenc_adv_e <- function(input_size, encoding_size,
                          encoder_hidden_sizes = c(60L, 60L),
                          decoder_hidden_sizes = c(60L, 60L),
                          discriminator_hidden_sizes = c(60L, 60L),
                          activation = c("relu", "leaky_relu", "elu", "gelu", "tanh"),
                          dropout = 0.4,
                          latent_prior_scale = 5,
                          lr_encoder = NULL,
                          lr_decoder = NULL,
                          lr_generator = NULL,
                          lr_discriminator = NULL,
                          batch_size = 350,
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
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$encoder_hidden_sizes <- normalize_hidden_sizes(encoder_hidden_sizes)
  obj$decoder_hidden_sizes <- normalize_hidden_sizes(decoder_hidden_sizes)
  obj$discriminator_hidden_sizes <- normalize_hidden_sizes(discriminator_hidden_sizes)
  obj$activation <- activation
  obj$dropout <- dropout
  obj$latent_prior_scale <- latent_prior_scale
  obj$lr_encoder <- lr_encoder
  obj$lr_decoder <- lr_decoder
  obj$lr_generator <- lr_generator
  obj$lr_discriminator <- lr_discriminator
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
  class(obj) <- append("autoenc_adv_e", class(obj))

  obj
}

#' @exportS3Method fit autoenc_adv_e
fit.autoenc_adv_e <- function(obj, data, ...) {
  if (!exists("autoenc_adv_create"))
    reticulate::source_python(system.file("python", "autoenc_adv.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- autoenc_adv_create(
      obj$input_size,
      obj$encoding_size,
      encoder_hidden_sizes = obj$encoder_hidden_sizes,
      decoder_hidden_sizes = obj$decoder_hidden_sizes,
      discriminator_hidden_sizes = obj$discriminator_hidden_sizes,
      activation = obj$activation,
      dropout = obj$dropout,
      latent_prior_scale = obj$latent_prior_scale,
      lr_encoder = obj$lr_encoder,
      lr_decoder = obj$lr_decoder,
      lr_generator = obj$lr_generator,
      lr_discriminator = obj$lr_discriminator,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  result <- autoenc_adv_fit(
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

#' @exportS3Method transform autoenc_adv_e
transform.autoenc_adv_e <- function(obj, data, ...) {
  if (!exists("autoenc_adv_create"))
    reticulate::source_python(system.file("python", "autoenc_adv.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_adv_encode(obj$model, data)
  }

  result
}