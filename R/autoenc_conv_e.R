#'@title Convolutional Autoencoder - Encode
#'@description Creates a deep learning convolutional autoencoder (ConvAE) to encode sequences
#' of observations. Wraps a PyTorch implementation.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return A `autoenc_conv_e` object.
#'
#'@references
#' Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011). Stacked Convolutional Auto-Encoders.
#'@examples
#'\dontrun{
#'# Conv1D-based encoder expects data reshaped internally to (n, input_size, 1)
#'X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#'ae <- autoenc_conv_e(input_size = 20, encoding_size = 5, num_epochs = 50)
#'ae <- daltoolbox::fit(ae, X)
#'Z  <- daltoolbox::transform(ae, X)   # 50 x 5 encodings
#'}
#'
#'# See:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/transf/autoenc_conv_e.md
#'@importFrom daltoolbox autoenc_base_e
#'@import reticulate
#'@export
autoenc_conv_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_conv_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_conv_e
fit.autoenc_conv_e <- function(obj, data, return_loss=FALSE, ...) {
  if (!exists("autoenc_conv_create"))
    reticulate::source_python(system.file("python", "autoenc_conv.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- autoenc_conv_create(obj$input_size, obj$encoding_size)

  result <- autoenc_conv_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@exportS3Method transform autoenc_conv_e
transform.autoenc_conv_e <- function(obj, data, ...) {
  if (!exists("autoenc_conv_create"))
    reticulate::source_python(system.file("python", "autoenc_conv.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_conv_encode(obj$model, data)
  }

  return(result)
}
