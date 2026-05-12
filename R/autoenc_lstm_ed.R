#' @title LSTM Autoencoder - Encode-Decode
#' @description Creates an LSTM-based autoencoder that reconstructs observations
#'   after sequence-aware compression through a Python/PyTorch backend.
#'
#' @inheritParams autoenc_lstm_e
#'
#' @return A `autoenc_lstm_ed` object.
#'
#' @references
#' Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#' ae <- autoenc_lstm_ed(
#'   input_size = 20,
#'   encoding_size = 5,
#'   lstm_hidden_size = 16,
#'   sequence_length = 4,
#'   num_layers = 2,
#'   dropout = 0.1
#' )
#' ae <- daltoolbox::fit(ae, X)
#' X_hat <- daltoolbox::transform(ae, X)
#' }
#'
#' @importFrom daltoolbox autoenc_base_ed
#' @import reticulate
#' @export
autoenc_lstm_ed <- function(input_size, encoding_size,
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
  obj <- daltoolbox::autoenc_base_ed(input_size, encoding_size)
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
  class(obj) <- append("autoenc_lstm_ed", class(obj))

  obj
}

#' @exportS3Method fit autoenc_lstm_ed
fit.autoenc_lstm_ed <- function(obj, data, ...) {
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

#' @exportS3Method transform autoenc_lstm_ed
transform.autoenc_lstm_ed <- function(obj, data, ...) {
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "autoenc_lstm.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_lstm_encode_decode(obj$model, data)
  }
  result
}