#'@title Adversarial Autoencoder - Encode-Decode
#'@description Creates a deep learning adversarial autoencoder (AAE) that encodes and decodes
#' sequences of observations. Wraps a PyTorch implementation.
#'
#'@details The adversarial loss constrains the latent distribution, improving sampling
#' and reconstruction quality in some setups compared to a vanilla AE.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@return A `autoenc_adv_ed` object.
#'
#'@references
#' Makhzani, A. et al. (2016). Adversarial Autoencoders.
#'
#'@examples
#'\dontrun{
#'X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#'ae <- autoenc_adv_ed(input_size = 20, encoding_size = 5, num_epochs = 50)
#'ae <- daltoolbox::fit(ae, X)
#'X_hat <- daltoolbox::transform(ae, X)  # reconstructions
#'mean((X - X_hat)^2)
#'}
#'
#'# More details:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_adv_ed.md
#'@importFrom daltoolbox autoenc_base_ed
#'@import reticulate
#'@export
autoenc_adv_ed <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001) {
  obj <- daltoolbox::autoenc_base_ed(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  class(obj) <- append("autoenc_adv_ed", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_adv_ed
fit.autoenc_adv_ed <- function(obj, data, ...){
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

#'@exportS3Method transform autoenc_adv_ed
transform.autoenc_adv_ed <- function(obj, data, ...) {
  if (!exists("autoenc_adv_create"))
    reticulate::source_python(system.file("python", "autoenc_adv.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_adv_encode_decode(obj$model, data)
  }

  return(result)
}
