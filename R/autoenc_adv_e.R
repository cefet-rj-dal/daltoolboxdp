#'@title Adversarial Autoencoder - Encode
#'@description Creates a deep learning adversarial autoencoder (AAE) to encode sequences
#' of observations. Wraps a PyTorch implementation.
#'
#'@details Adversarial autoencoders regularize the latent space using an adversarial
#' training objective, encouraging the aggregated posterior to match a prior distribution.
#' This can lead to more structured latent representations.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return A `autoenc_adv_e` object.
#'
#'@references
#' Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2016).
#' Adversarial Autoencoders.
#'
#'@examples
#'\dontrun{
#'X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#'ae <- autoenc_adv_e(input_size = 20, encoding_size = 5, num_epochs = 50)
#'ae <- daltoolbox::fit(ae, X)
#'Z  <- daltoolbox::transform(ae, X)
#'}
#'
#'# See a complete example:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_adv_e.md
#'@importFrom daltoolbox autoenc_base_e
#'@import reticulate
#'@export
autoenc_adv_e <- function(input_size, encoding_size, batch_size = 350, num_epochs = 1000, learning_rate = 0.001) {
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_adv_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_adv_e
fit.autoenc_adv_e <- function(obj, data, ...) {
  if (!exists("autoenc_adv_create"))
    reticulate::source_python(system.file("python", "autoenc_adv.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- autoenc_adv_create(obj$input_size, obj$encoding_size)

  result <- autoenc_adv_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@exportS3Method transform autoenc_adv_e
transform.autoenc_adv_e <- function(obj, data, ...) {
  if (!exists("autoenc_adv_create"))
    reticulate::source_python(system.file("python", "autoenc_adv.py", package = "daltoolboxdp"))

  result <- NULL

  if (!is.null(obj$model)) {
    result <- autoenc_adv_encode(obj$model, data)
  }

  return(result)
}
