#'@title Autoencoder - Encode
#'@description Creates a deep learning autoencoder that learns a latent representation (encoding)
#' for a sequence of observations. Wraps a PyTorch implementation via the `reticulate` bridge.
#'
#'@details This encoder provides dimensionality reduction by training a neural network to
#' compress inputs into a lower-dimensional bottleneck. The learned encoding can be used
#' for downstream tasks such as clustering, visualization, or as features for predictive models.
#'
#'@param input_size Integer. Number of input features per observation.
#'@param encoding_size Integer. Size of the latent (bottleneck) representation.
#'@param batch_size Integer. Mini-batch size used during training. Default is 32.
#'@param num_epochs Integer. Maximum number of training epochs. Default is 100.
#'@param learning_rate Numeric. Optimizer learning rate. Default is 0.001.
#'@param validation_strategy Character. One of `static` or `dynamic`.
#'@param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#'@param val_ratio Numeric. Validation fraction used when validation is enabled. Default is 0.3.
#'@param patience Integer. Early stopping patience. Default is 100.
#'@param min_delta Numeric. Minimum improvement to reset early stopping. Default is 1e-4.
#'@param sma_window Integer. Window size used by `sma`. Default is 5.
#'@param ema_alpha Numeric. Smoothing factor used by `ema`. Default is 0.2.
#'@param test_window Integer. Window size used by `h`. Default is 30.
#'@param p_value Numeric. Significance threshold used by `h`. Default is 0.05.
#'@param seed Integer. Seed used by data splitting routines. Default is 42.
#'
#'@return A `autoenc_e` object.
#'
#'@references
#' Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks.
#' Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
#'
#'@examples
#'\dontrun{
#'# Requirements: Python with torch installed and reticulate configured.
#'set.seed(123)
#'
#'# 1) Create a toy dataset with 100 samples and 20 features
#'X <- matrix(rnorm(2000), nrow = 100, ncol = 20)
#'
#'# 2) Create and fit an encoder (5-D bottleneck)
#'ae <- autoenc_e(input_size = 20, encoding_size = 5, num_epochs = 100)
#'ae <- daltoolbox::fit(ae, X)
#'
#'# 3) Transform data to latent space
#'Z <- daltoolbox::transform(ae, X)   # matrix with dimensions 100 x 5
#'dim(Z)                              # c(100, 5)
#'}
#'
#'# See a more complete example at:
#'# https://github.com/cefet-rj-dal/daltoolbox/blob/main/autoencoder/autoenc_e.md
#'
#'@importFrom daltoolbox autoenc_base_e
#'@import reticulate
#'@export
autoenc_e <- function(input_size, encoding_size, batch_size = 32, num_epochs = 100L, learning_rate = 0.001,
                      validation_strategy = c("static", "dynamic"),
                      stopping_rule = c("none", "patience", "sma", "ema", "h"),
                      val_ratio = 0.3, patience = 100L, min_delta = 1e-4,
                      sma_window = 5L, ema_alpha = 0.2, test_window = 30L, p_value = 0.05, seed = 42L) {
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- daltoolbox::autoenc_base_e(input_size, encoding_size)
  obj$input_size <- input_size
  obj$encoding_size <- encoding_size
  obj$batch_size <- batch_size
  obj$num_epochs <- num_epochs
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
  obj$seed <- seed
  class(obj) <- append("autoenc_e", class(obj))

  return(obj)
}

#'@exportS3Method fit autoenc_e
fit.autoenc_e <- function(obj, data, ...) {
  # Lazily load Python implementation only when needed
  if (!exists("autoenc_create"))
    reticulate::source_python(system.file("python", "autoenc.py", package = "daltoolboxdp"))

  # Create model once and reuse it across fits
  if (is.null(obj$model))
    obj$model <- autoenc_create(obj$input_size, obj$encoding_size, validation_strategy = obj$validation_strategy, stopping_rule = obj$stopping_rule)

  # Train and capture training/validation losses
  result <- autoenc_fit(obj$model, data, batch_size = obj$batch_size, num_epochs = obj$num_epochs, learning_rate = obj$learning_rate,
                        validation_strategy = obj$validation_strategy, stopping_rule = obj$stopping_rule, val_ratio = obj$val_ratio,
                        patience = obj$patience, min_delta = obj$min_delta, sma_window = obj$sma_window, ema_alpha = obj$ema_alpha,
                        test_window = obj$test_window, p_value = obj$p_value, seed = obj$seed)

  obj$model <- result[[1]]
  obj$train_loss <- result[[2]]
  obj$val_loss <- result[[3]]

  return(obj)
}

#'@exportS3Method transform autoenc_e
transform.autoenc_e <- function(obj, data, ...) {
  # Ensure backing Python functions are available
  if (!exists("autoenc_create"))
    reticulate::source_python(system.file("python", "autoenc.py", package = "daltoolboxdp"))

  result <- NULL
  if (!is.null(obj$model)) {
    # Forward pass to obtain latent encodings
    result <- autoenc_encode(obj$model, data)
  }
  return(result)
}
