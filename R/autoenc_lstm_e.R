#' @title LSTM Autoencoder - Encode
#' @description Creates an LSTM-based autoencoder with configurable recurrent depth
#'   and latent projection through a Python/PyTorch backend.
#'
#' @details
#' `encoding_size` remains the latent bottleneck exposed to the user. The recurrent
#' body can now use a different `lstm_hidden_size`, multiple layers, dropout between
#' recurrent layers, and a configurable `sequence_length` to reshape each row into
#' a sequence before encoding.
#'
#' @param input_size Integer. Number of input features per observation.
#' @param encoding_size Integer. Size of the latent (bottleneck) representation.
#' @param lstm_hidden_size Optional integer. Hidden size used inside the encoder/decoder LSTMs.
#'   If `NULL`, it defaults to `encoding_size`.
#' @param sequence_length Integer. Number of time steps represented by each row. `input_size`
#'   must be divisible by `sequence_length`. Default is `1L`, which preserves the previous behavior.
#' @param num_layers Integer. Number of recurrent LSTM layers.
#' @param dropout Numeric. Recurrent dropout applied between LSTM layers when `num_layers > 1`.
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
#' @return A `autoenc_lstm_e` object.
#'
#' @references
#' Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_lstm_e(
#'   input_size = 20,
#'   encoding_size = 5,
#'   lstm_hidden_size = 16,
#'   sequence_length = 4,
#'   num_layers = 2,
#'   dropout = 0.1
#' )
#' ae <- daltoolbox::fit(ae, X)
#' Z <- daltoolbox::transform(ae, X)
#' dim(Z)
#' }
#'
#' @importFrom daltoolbox autoenc_base_e
#' @import reticulate
#' @export
autoenc_lstm_e <- function(input_size, encoding_size,
                           lstm_hidden_size = NULL,
                           sequence_length = 1L,
                           num_layers = 1L,
                           dropout = 0,
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
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$lstm_hidden_size <- lstm_hidden_size
  obj$sequence_length <- sequence_length
  obj$num_layers <- num_layers
  obj$dropout <- dropout
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
  class(obj) <- append("autoenc_lstm_e", class(obj))

  obj
}

#' @exportS3Method fit autoenc_lstm_e
fit.autoenc_lstm_e <- function(obj, data, ...) {
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "autoenc_lstm.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- autoenc_lstm_create(
      obj$input_size,
      obj$encoding_size,
      lstm_hidden_size = obj$lstm_hidden_size,
      sequence_length = obj$sequence_length,
      num_layers = obj$num_layers,
      dropout = obj$dropout,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  result <- autoenc_lstm_fit(
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

#' @exportS3Method transform autoenc_lstm_e
transform.autoenc_lstm_e <- function(obj, data, ...) {
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "autoenc_lstm.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_lstm_encode(obj$model, data)
  }
  result
}