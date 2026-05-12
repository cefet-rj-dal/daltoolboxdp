#' @title PyTorch Time-Series MLP
#' @description Time-series forecaster using a configurable feedforward PyTorch MLP with unified
#'   training strategies and a Python backend.
#'
#' @param preprocess Optional preprocessing/normalization object.
#' @param input_size Integer. Number of lagged inputs per training example.
#' @param hidden_sizes Integer vector with hidden layer sizes.
#' @param dropout Numeric. Dropout rate.
#' @param activation Character. Hidden activation function. One of
#'   `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, or `"tanh"`.
#' @param output_activation Character. Output activation of the regression head. One of
#'   `"none"`, `"relu"`, `"sigmoid"`, `"tanh"`, or `"softplus"`.
#' @param normalization Character. Optional normalization after each hidden linear layer.
#'   One of `"none"`, `"batch"`, or `"layer"`.
#' @param init_method Character. Weight initialization strategy. One of
#'   `"default"`, `"xavier_uniform"`, `"xavier_normal"`, `"kaiming_uniform"`, or `"kaiming_normal"`.
#' @param epochs Integer. Maximum number of training epochs. Default is `100L`.
#' @param lr Numeric. Optimizer learning rate.
#' @param validation_strategy Character. One of `static` or `dynamic`.
#' @param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#' @param val_ratio Numeric. Validation fraction used when validation is enabled.
#' @param batch_size Integer. Mini-batch size.
#' @param patience Integer. Early stopping patience.
#' @param min_delta Numeric. Minimum improvement to reset early stopping.
#' @param sma_window Integer. Window size used by `sma`.
#' @param ema_alpha Numeric. Smoothing factor used by `ema`.
#' @param test_window Integer. Window size used by `h`.
#' @param p_value Numeric. Significance threshold used by `h`.
#' @return A `torch_ts_mlp` object.
#' @examples
#' \dontrun{
#' library(daltoolboxdp)
#' model <- torch_ts_mlp(
#'   input_size = 12,
#'   hidden_sizes = c(32L, 16L),
#'   normalization = "batch",
#'   init_method = "kaiming_uniform",
#'   epochs = 100L
#' )
#' }
#' @importFrom tspredit ts_regsw
#' @import reticulate
#' @export
torch_ts_mlp <- function(preprocess = NA,
                         input_size = NA,
                         hidden_sizes = c(16L, 8L),
                         dropout = 0,
                         activation = c("relu", "leaky_relu", "elu", "gelu", "tanh"),
                         output_activation = c("none", "relu", "sigmoid", "tanh", "softplus"),
                         normalization = c("none", "batch", "layer"),
                         init_method = c("default", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"),
                         epochs = 100L,
                         lr = 0.001,
                         validation_strategy = c("static", "dynamic"),
                         stopping_rule = c("none", "patience", "sma", "ema", "h"),
                         val_ratio = 0.2,
                         batch_size = 32L,
                         patience = 100L,
                         min_delta = 1e-4,
                         sma_window = 5L,
                         ema_alpha = 0.2,
                         test_window = 30L,
                         p_value = 0.05) {
  activation <- match.arg(activation)
  output_activation <- match.arg(output_activation)
  normalization <- match.arg(normalization)
  init_method <- match.arg(init_method)
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)

  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$hidden_sizes <- as.integer(hidden_sizes)
  obj$dropout <- as.numeric(dropout)
  obj$activation <- activation
  obj$output_activation <- output_activation
  obj$normalization <- normalization
  obj$init_method <- init_method
  obj$epochs <- as.integer(epochs)
  obj$lr <- as.numeric(lr)
  obj$validation_strategy <- validation_strategy
  obj$stopping_rule <- stopping_rule
  obj$val_ratio <- as.numeric(val_ratio)
  obj$batch_size <- as.integer(batch_size)
  obj$patience <- as.integer(patience)
  obj$min_delta <- as.numeric(min_delta)
  obj$sma_window <- as.integer(sma_window)
  obj$ema_alpha <- as.numeric(ema_alpha)
  obj$test_window <- as.integer(test_window)
  obj$p_value <- as.numeric(p_value)
  class(obj) <- append("torch_ts_mlp", class(obj))

  obj
}

#' @importFrom tspredit do_fit
#' @exportS3Method do_fit torch_ts_mlp
do_fit.torch_ts_mlp <- function(obj, x, y) {
  if (!exists("torch_ts_mlp_create"))
    reticulate::source_python(system.file("python", "torch_ts_mlp.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- torch_ts_mlp_create(
      obj$input_size,
      obj$hidden_sizes,
      dropout = obj$dropout,
      activation = obj$activation,
      output_activation = obj$output_activation,
      normalization = obj$normalization,
      init_method = obj$init_method,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- torch_ts_mlp_fit(
    obj$model,
    df_train,
    epochs = obj$epochs,
    lr = obj$lr,
    validation_strategy = obj$validation_strategy,
    stopping_rule = obj$stopping_rule,
    val_ratio = obj$val_ratio,
    batch_size = obj$batch_size,
    patience = obj$patience,
    min_delta = obj$min_delta,
    sma_window = obj$sma_window,
    ema_alpha = obj$ema_alpha,
    test_window = obj$test_window,
    p_value = obj$p_value
  )

  obj$train_loss_hist <- obj$model$train_loss_hist
  obj$val_loss_hist <- obj$model$val_loss_hist
  obj$epochs_done <- obj$model$epochs_done
  obj
}

#' @importFrom tspredit do_predict
#' @exportS3Method do_predict torch_ts_mlp
do_predict.torch_ts_mlp <- function(obj, x) {
  if (!exists("torch_ts_mlp_predict"))
    reticulate::source_python(system.file("python", "torch_ts_mlp.py", package = "daltoolboxdp"))

  x_values <- as.data.frame(x)
  x_values$t0 <- 0
  torch_ts_mlp_predict(obj$model, x_values, batch_size = obj$batch_size)
}