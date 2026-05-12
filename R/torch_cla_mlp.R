#' @title PyTorch MLP Classifier
#' @description Classification model backed by a configurable PyTorch MLP with unified training strategies.
#' @param attribute Target attribute name.
#' @param slevels Vector with valid class labels.
#' @param preprocess Optional preprocessing object.
#' @param input_size Integer. Number of input attributes.
#' @param hidden_sizes Integer vector with hidden layer sizes.
#' @param num_classes Integer. Number of classes. Defaults to `length(slevels)`.
#' @param dropout Numeric. Dropout rate.
#' @param activation Character. Hidden activation function. One of
#'   `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, or `"tanh"`.
#' @param normalization Character. Optional normalization after each hidden linear layer.
#'   One of `"none"`, `"batch"`, or `"layer"`.
#' @param init_method Character. Weight initialization strategy. One of
#'   `"default"`, `"xavier_uniform"`, `"xavier_normal"`, `"kaiming_uniform"`, or `"kaiming_normal"`.
#' @param epochs Integer. Maximum number of epochs. Default is `100L`.
#' @param lr Numeric. Learning rate.
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
#' @param weight_decay Numeric. L2 regularization.
#' @examples
#' \dontrun{
#' library(daltoolboxdp)
#' model <- torch_cla_mlp(
#'   attribute = "class",
#'   slevels = c("A", "B"),
#'   input_size = 10,
#'   hidden_sizes = c(64L, 32L),
#'   normalization = "batch",
#'   init_method = "kaiming_uniform",
#'   epochs = 1000L
#' )
#' }
#' @import daltoolbox
#' @import reticulate
#' @export
torch_cla_mlp <- function(attribute,
                          slevels,
                          preprocess = NA,
                          input_size,
                          hidden_sizes,
                          num_classes = length(slevels),
                          dropout = 0,
                          activation = c("relu", "leaky_relu", "elu", "gelu", "tanh"),
                          normalization = c("none", "batch", "layer"),
                          init_method = c("default", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"),
                          epochs = 100L,
                          lr = 1e-3,
                          validation_strategy = c("static", "dynamic"),
                          stopping_rule = c("none", "patience", "sma", "ema", "h"),
                          val_ratio = 0.2,
                          batch_size = 64L,
                          patience = 100L,
                          min_delta = 1e-4,
                          sma_window = 5L,
                          ema_alpha = 0.2,
                          test_window = 30L,
                          p_value = 0.05,
                          weight_decay = 0) {
  activation <- match.arg(activation)
  normalization <- match.arg(normalization)
  init_method <- match.arg(init_method)
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    preprocess = preprocess,
    input_size = as.integer(input_size),
    hidden_sizes = as.integer(hidden_sizes),
    num_classes = as.integer(num_classes),
    dropout = as.numeric(dropout),
    activation = activation,
    normalization = normalization,
    init_method = init_method,
    epochs = as.integer(epochs),
    lr = as.numeric(lr),
    validation_strategy = validation_strategy,
    stopping_rule = stopping_rule,
    val_ratio = as.numeric(val_ratio),
    batch_size = as.integer(batch_size),
    patience = as.integer(patience),
    min_delta = as.numeric(min_delta),
    sma_window = as.integer(sma_window),
    ema_alpha = as.numeric(ema_alpha),
    test_window = as.integer(test_window),
    p_value = as.numeric(p_value),
    weight_decay = as.numeric(weight_decay),
    model = NULL,
    classes_ = NULL
  )
  obj <- c(obj, objex)
  class(obj) <- c("torch_cla_mlp", cobj)
  obj
}

#' @exportS3Method fit torch_cla_mlp
fit.torch_cla_mlp <- function(obj, data, ...) {
  if (!exists("torch_cla_mlp_create"))
    reticulate::source_python(system.file("python", "torch_cla_mlp.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- torch_cla_mlp_create(
      obj$input_size,
      obj$hidden_sizes,
      obj$num_classes,
      dropout = obj$dropout,
      activation = obj$activation,
      normalization = obj$normalization,
      init_method = obj$init_method,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  df_train <- adjust_data.frame(data)
  df_train[, obj$attribute] <- adjust_factor(df_train[, obj$attribute], obj$ilevels, obj$slevels)
  obj$x <- setdiff(colnames(df_train), obj$attribute)

  obj$model <- torch_cla_mlp_fit(
    obj$model,
    df_train,
    target_column = obj$attribute,
    epochs = obj$epochs,
    lr = obj$lr,
    validation_strategy = obj$validation_strategy,
    stopping_rule = obj$stopping_rule,
    batch_size = obj$batch_size,
    val_ratio = obj$val_ratio,
    patience = obj$patience,
    min_delta = obj$min_delta,
    sma_window = obj$sma_window,
    ema_alpha = obj$ema_alpha,
    test_window = obj$test_window,
    p_value = obj$p_value,
    weight_decay = obj$weight_decay,
    classes_ = obj$slevels
  )

  obj$classes_ <- obj$model$classes_
  obj$train_loss_hist <- obj$model$train_loss_hist
  obj$val_loss_hist <- obj$model$val_loss_hist
  obj$epochs_done <- obj$model$epochs_done
  obj
}

#' @export
predict.torch_cla_mlp <- function(object, x, ...) {
  if (!exists("torch_cla_mlp_predict_scores"))
    reticulate::source_python(system.file("python", "torch_cla_mlp.py", package = "daltoolboxdp"))

  x <- adjust_data.frame(x)
  x <- x[, object$x, drop = FALSE]
  prediction <- torch_cla_mlp_predict_scores(object$model, as.data.frame(x), object$classes_)
  prediction <- do.call(rbind, prediction)
  prediction <- as.data.frame(prediction)
  colnames(prediction) <- object$slevels
  prediction
}