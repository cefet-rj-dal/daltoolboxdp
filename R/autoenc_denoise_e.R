#'@title Denoising Autoencoder - Encode
#'@description Creates a deep learning denoising autoencoder (DAE) to encode sequences
#' while learning robustness to noise. Wraps a PyTorch implementation.
#'@param input_size input size
#'@param encoding_size encoding size
#'@param batch_size size for batch learning
#'@param num_epochs number of epochs for training
#'@param learning_rate learning rate
#'@param noise_factor Numeric. Standard deviation (scale) of the noise added during training.
#'@return A `autoenc_denoise_e` object.
#'
#'@references
#' Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and Composing Robust Features with Denoising Autoencoders.
#'
#'@examples
#'\dontrun{
#'# 1) Prepare data
#'X <- matrix(rnorm(1000), nrow = 50, ncol = 20)
#'
#'# 2) Fit denoising encoder (higher noise_factor = stronger noise during training)
#'ae <- autoenc_denoise_e(input_size = 20, encoding_size = 5, noise_factor = 0.2, num_epochs = 50)
#'ae <- daltoolbox::fit(ae, X)
#'
#'# 3) Obtain latent encodings
#'Z <- daltoolbox::transform(ae, X)
#'dim(Z)
#'}
#'
#'# See:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_denoise_e.md
#'@importFrom daltoolbox autoenc_base_e
#'@import reticulate
#'@export
autoenc_denoise_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, noise_factor=0.3) {
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
  obj$learning_rate <- learning_rate
  obj$noise_factor <- noise_factor
  class(obj) <- append("autoenc_denoise_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_denoise_e
fit.autoenc_denoise_e <- function(obj, data, ...) {
  if (!exists("autoenc_denoise_create"))
    reticulate::source_python(system.file("python", "autoenc_denoise.py", package = "daltoolboxdp"))

  if (is.null(obj$model))
    obj$model <- autoenc_denoise_create(obj$input_size, obj$encoding_size, noise_factor = obj$noise_factor)

  result <- autoenc_denoise_fit(obj$model, data, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]


  return(obj)
}

#'@exportS3Method transform autoenc_denoise_e
transform.autoenc_denoise_e <- function(obj, data, ...) {
  if (!exists("autoenc_denoise_create"))
    reticulate::source_python(system.file("python", "autoenc_denoise.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    result <- autoenc_denoise_encode(obj$model, data)
  }
  return(result)
}
