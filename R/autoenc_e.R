#'@title Autoencoder - Encode
#'@description Creates an deep learning autoencoder to encode a sequence of observations.
#' It wraps the pytorch and reticulate libraries.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return returns a `autoenc_e` object.
#'@examples
#'#See an example of using `autoenc_e` at this
#'#https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_e.md
#'@importFrom daltoolbox autoenc_base_e
#'@import reticulate
#'@export
autoenc_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_e
fit.autoenc_e <- function(obj, data, ...) {
  if (!exists("autoenc_create"))
    reticulate::source_python(system.file("python", "autoenc.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- autoenc_create(obj$input_size, obj$encoding_size)

  result <- autoenc_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@exportS3Method transform autoenc_e
transform.autoenc_e <- function(obj, data, ...) {
  if (!exists("autoenc_create"))
    reticulate::source_python(system.file("python", "autoenc.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_encode(obj$model, data)
  }
  return(result)
}
