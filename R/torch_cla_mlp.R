#'@title PyTorch MLP Classifier
#'@description Classification model backed by a PyTorch MLP with unified training strategies.
#'@param attribute Target attribute name.
#'@param slevels Vector with valid class labels.
#'@param preprocess Optional preprocessing object.
#'@param input_size Integer. Number of input attributes.
#'@param hidden_sizes Integer vector with hidden layer sizes.
#'@param num_classes Integer. Number of classes. Defaults to `length(slevels)`.
#'@param dropout Numeric. Dropout rate.
#'@param epochs Integer. Maximum number of epochs. Default is `100L`.
#'@param lr Numeric. Learning rate.
#'@param validation_strategy Character. One of `static` or `dynamic`.
#'@param stopping_rule Character. One of `none`, `patience`, `sma`, `ema`, or `h`.
#'@param val_ratio Numeric. Validation fraction used when validation is enabled.
#'@param batch_size Integer. Mini-batch size.
#'@param patience Integer. Early stopping patience.
#'@param min_delta Numeric. Minimum improvement to reset early stopping.
#'@param sma_window Integer. Window size used by `sma`.
#'@param ema_alpha Numeric. Smoothing factor used by `ema`.
#'@param test_window Integer. Window size used by `h`.
#'@param p_value Numeric. Significance threshold used by `h`.
#'@param weight_decay Numeric. L2 regularization.
#'@param seed Integer. Seed used by data splitting routines.
#'@examples
#'\dontrun{
#'library(daltoolboxdp)
#'model <- torch_cla_mlp(
#'  attribute = "class",
#'  slevels = c("A", "B"),
#'  input_size = 10,
#'  hidden_sizes = c(64L, 32L),
#'  epochs = 100L
#')
#'}
#'@import daltoolbox
#'@import reticulate
#'@export
torch_cla_mlp <- function(attribute,
                          slevels,
                          preprocess = NA,
                          input_size,
                          hidden_sizes,
                          num_classes = length(slevels),
                          dropout = 0,
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
                          weight_decay = 0,
                          seed = 42L) {
  validation_strategy <- match.arg(validation_strategy)
  stopping_rule <- match.arg(stopping_rule)
  obj <- classification(attribute, slevels)
  objex <- list(
    preprocess = preprocess,
    input_size = as.integer(input_size),
    hidden_sizes = as.integer(hidden_sizes),
    num_classes = as.integer(num_classes),
    dropout = as.numeric(dropout),
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
    seed = if (is.null(seed)) NULL else as.integer(seed),
    model = NULL,
    classes_ = NULL
  )
  obj <- c(obj, objex)
  class(obj) <- append("torch_cla_mlp", class(obj))
  obj
}

#'@exportS3Method fit torch_cla_mlp
fit.torch_cla_mlp <- function(obj, data, ...) {
  if (!exists("torch_cla_mlp_create"))
    reticulate::source_python(system.file("python", "torch_cla_mlp.py", package = "daltoolboxdp"))

  if (is.null(obj$model)) {
    obj$model <- torch_cla_mlp_create(
      obj$input_size,
      obj$hidden_sizes,
      obj$num_classes,
      dropout = obj$dropout,
      validation_strategy = obj$validation_strategy,
      stopping_rule = obj$stopping_rule
    )
  }

  df_train <- as.data.frame(data)
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
    classes_ = obj$slevels,
    seed = obj$seed
  )

  obj$classes_ <- obj$model$classes_
  obj$train_loss_hist <- obj$model$train_loss_hist
  obj$val_loss_hist <- obj$model$val_loss_hist
  obj$epochs_done <- obj$model$epochs_done
  obj
}

#'@export
predict.torch_cla_mlp <- function(object, x, ...) {
  if (!exists("torch_cla_mlp_predict"))
    reticulate::source_python(system.file("python", "torch_cla_mlp.py", package = "daltoolboxdp"))

  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute, drop = FALSE]
  prediction <- torch_cla_mlp_predict(object$model, as.data.frame(x), object$classes_)
  adjust_class_label(prediction)
}

#'@rdname torch_cla_mlp
#'@param obj Fitted `torch_cla_mlp` model.
#'@param x Data frame or matrix with predictor columns.
#'@export
predict_proba.torch_cla_mlp <- function(obj, x) {
  if (!exists("torch_cla_mlp_predict_proba"))
    reticulate::source_python(system.file("python", "torch_cla_mlp.py", package = "daltoolboxdp"))

  df_test <- as.data.frame(x)
  torch_cla_mlp_predict_proba(obj$model, df_test)
}
