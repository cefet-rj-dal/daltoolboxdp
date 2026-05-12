#' Neural Network Classifier
#' @title Multi-layer Perceptron Classifier
#' @description Implements classification using a multi-layer perceptron (MLP).
#' Wraps scikit-learn's `MLPClassifier` through `reticulate`.
#' @param attribute Target attribute name for model building.
#' @param slevels List of possible values for classification target.
#' @param hidden_layer_sizes Number of neurons in each hidden layer.
#' @param activation Activation function for hidden layers. One of
#'   `"relu"`, `"identity"`, `"logistic"`, or `"tanh"`.
#' @param solver Optimizer used for training. One of `"adam"`, `"lbfgs"`, or `"sgd"`.
#' @param alpha L2 penalty (regularization term).
#' @param batch_size Size of minibatches for stochastic optimizers. Use `"auto"` or an integer.
#' @param learning_rate_init Initial learning rate used by stochastic solvers.
#' @param max_iter Maximum number of iterations.
#' @param early_stopping Whether to use early stopping.
#' @return A `skcla_mlp` classifier object.
#'
#' @references
#' Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
#' @examples
#' \dontrun{
#' data(iris)
#' clf <- skcla_mlp(
#'   attribute = "Species",
#'   slevels = levels(iris$Species),
#'   hidden_layer_sizes = c(32, 16),
#'   activation = "relu"
#' )
#' clf <- daltoolbox::fit(clf, iris)
#' pred <- predict(clf, iris)
#' table(pred, iris$Species)
#' }
#' @import daltoolbox
#' @export
skcla_mlp <- function(attribute, slevels,
                      hidden_layer_sizes = c(100),
                      activation = c("relu", "identity", "logistic", "tanh"),
                      solver = c("adam", "lbfgs", "sgd"),
                      alpha = 0.0001,
                      batch_size = "auto",
                      learning_rate_init = 0.001,
                      max_iter = 200,
                      early_stopping = FALSE) {
  activation <- match.arg(activation)
  solver <- match.arg(solver)

  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    hidden_layer_sizes = as.integer(hidden_layer_sizes),
    activation = activation,
    solver = solver,
    alpha = as.numeric(alpha),
    batch_size = batch_size,
    learning_rate_init = as.numeric(learning_rate_init),
    max_iter = as.integer(max_iter),
    early_stopping = early_stopping
  )

  obj <- c(obj, objex)
  class(obj) <- c("skcla_mlp", cobj)
  obj
}

#' @import daltoolbox
#' @import reticulate
#' @exportS3Method fit skcla_mlp
fit.skcla_mlp <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_mlp.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)

  if (is.null(obj$model)) {
    obj$model <- skcla_mlp_create(
      hidden_layer_sizes = obj$hidden_layer_sizes,
      activation = obj$activation,
      solver = obj$solver,
      alpha = obj$alpha,
      batch_size = obj$batch_size,
      learning_rate_init = obj$learning_rate_init,
      max_iter = obj$max_iter,
      early_stopping = obj$early_stopping
    )
  }

  prepared <- prepare_skcla_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  obj$model <- skcla_mlp_fit(obj$model, data, obj$attribute)

  obj
}

#' @import daltoolbox
#' @import reticulate
#' @export
predict.skcla_mlp <- function(object, x, ...) {
  if (!exists("skcla_mlp_predict_proba")) {
    python_path <- system.file("python/skcla_mlp.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }

  x <- prepare_skcla_predict_data(object, x)

  prediction <- skcla_mlp_predict_proba(object$model, x)
  prediction <- skcla_as_probability(prediction, object$slevels, object$model$classes_)

  prediction
}
