#'@title LSTM Autoencoder - Encode-Decode
#'@description Creates a deep learning LSTM-based autoencoder that encodes and decodes
#' sequences of observations. Wraps a PyTorch implementation via `reticulate`.
#'
#'@param input_size Integer. Number of input features per observation.
#'@param encoding_size Integer. Size of the latent (bottleneck) representation.
#'@param batch_size Integer. Mini-batch size used during training. Default is 32.
#'@param num_epochs Integer. Maximum number of training epochs. Default is 50.
#'@param learning_rate Numeric. Optimizer learning rate. Default is 0.001.
#'
#'@return A `autoenc_lstm_ed` object.
#'
#'@references
#' Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
#'
#'@examples
#'\dontrun{
#'X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#'ae <- autoenc_lstm_ed(input_size = 20, encoding_size = 5, num_epochs = 50)
#'ae <- daltoolbox::fit(ae, X)
#'X_hat <- daltoolbox::transform(ae, X)  # reconstructions
#'mean((X - X_hat)^2)
#'}
#'
#'# See:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_lstm_ed.md
#'@importFrom daltoolbox autoenc_base_ed
#'@import reticulate
#'@export
autoenc_lstm_ed <- function(input_size, encoding_size, batch_size = 32, num_epochs = 50, learning_rate = 0.001) {
  obj <- daltoolbox::autoenc_base_ed(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_lstm_ed", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_lstm_ed
fit.autoenc_lstm_ed <- function(obj, data, ...) {
  # Lazily load Python implementation only when needed
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "autoenc_lstm.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- autoenc_lstm_create(obj$input_size, obj$encoding_size)

  # Train and collect loss traces
  result <- autoenc_lstm_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@exportS3Method transform autoenc_lstm_ed
transform.autoenc_lstm_ed <- function(obj, data, ...) {
  # Ensure Python functions are available
  if (!exists("autoenc_lstm_create"))
    reticulate::source_python(system.file("python", "autoenc_lstm.py", package = "daltoolboxdp"))

  result <- NULL

  if (!is.null(obj$model)) {
    # Reconstruction pass using the trained LSTM autoencoder
    result <- autoenc_lstm_encode_decode(obj$model, data)
  }
  return(result)
}
