#' @title Variational Autoencoder - Encode-Decode
#' @description Creates a variational autoencoder (VAE) that reconstructs observations
#'   from a probabilistic latent space through a Python/PyTorch backend.
#'
#' @inheritParams autoenc_variational_e
#'
#' @return A `autoenc_variational_ed` object.
#'
#' @references
#' Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_variational_ed(
#'   input_size = 20,
#'   encoding_size = 5,
#'   encoder_hidden_sizes = c(128L, 64L, 32L),
#'   reconstruction_loss = "mse"
#' )
#' ae <- daltoolbox::fit(ae, X)
#' X_hat <- daltoolbox::transform(ae, X)
#' }
#'
#' @importFrom daltoolbox autoenc_base_ed
#' @import reticulate
#' @export
autoenc_variational_ed <- function(input_size, encoding_size,
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
  obj <- daltoolbox::autoenc_base_ed(input_size, encoding_size)
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
  class(obj) <- append("autoenc_variational_ed", class(obj))

  obj
}

#' @exportS3Method fit autoenc_variational_ed
fit.autoenc_variational_ed <- function(obj, data, ...) {
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

#' @exportS3Method transform autoenc_variational_ed
transform.autoenc_variational_ed <- function(obj, data, ...) {
  if (!exists("autoenc_variational_create"))
    reticulate::source_python(system.file("python", "autoenc_variational.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_variational_encode_decode(obj$model, data)
  }
  result
}