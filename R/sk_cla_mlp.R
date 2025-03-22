#' Neural Network
#' 
#' Deep learning classification
#' @import reticulate
#' @import daltoolbox
#' @title Multi-Layer Perceptron Classifier
#' @description Neural network classifier
#' @param attribute Target variable
#' @param slevels Possible values for the target classification.
#' @param hidden_layer_sizes The ith element represents the number of neurons in the ith hidden layer.
#' @param activation Activation function for the hidden layer ('identity', 'logistic', 'tanh', 'relu').
#' @param solver The solver for weight optimization ('lbfgs', 'sgd', 'adam').
#' @param alpha L2 penalty (regularization term) parameter.
#' @param batch_size Size of minibatches for stochastic optimizers.
#' @param learning_rate Schedule for weight updates ('constant', 'invscaling', 'adaptive').
#' @param max_iter Maximum number of iterations.
#' @param tol Tolerance for optimization termination.
#' @param random_state Seed for random number generation.
#' @return A mlp object.
#' @export
cla_mlp <- function(attribute, slevels,
                    hidden_layer_sizes = c(100),
                    activation = "relu",
                    solver = "adam",
                    alpha = 0.0001,
                    batch_size = "auto",
                    learning_rate = "constant",
                    max_iter = 200,
                    tol = 0.0001,
                    random_state = NULL) {
  obj <- list(
    attribute = attribute,
    slevels = slevels,
    hidden_layer_sizes = as.integer(hidden_layer_sizes),
    activation = activation,
    solver = solver,
    alpha = as.numeric(alpha),
    batch_size = batch_size,
    learning_rate = learning_rate,
    max_iter = as.integer(max_iter),
    tol = as.numeric(tol),
    random_state = if(!is.null(random_state)) as.integer(random_state) else NULL
  )
  
  class(obj) <- c("cla_mlp", class(obj))
  return(obj)
}

#' @import reticulate
#' @export
#'@method fit cla_mlp
#'@param obj A Neural Network classifier object
#'@param data Input data frame containing features and target variable
#'@param ... Additional arguments passed to the function
#'@return A fitted Neural Network classifier object
#'@export
fit.cla_mlp <- function(obj, data, ...) {
  if (!exists("mlp_create")) {
    python_path <- system.file("python/sklearn/cla_mlp.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  if (is.null(obj$model)) {
    obj$model <- mlp_create(
      obj$hidden_layer_sizes,
      obj$activation,
      obj$solver,
      obj$alpha,
      obj$batch_size,
      obj$learning_rate,
      obj$max_iter,
      obj$tol,
      obj$random_state
    )
  }
  
  data <- adjust_data.frame(data)
  obj$model <- mlp_fit(obj$model, data, obj$attribute)
  
  return(obj)
}

#' @import reticulate
#' @export
predict.cla_mlp <- function(obj, data, ...) {
  if (!exists("mlp_predict")) {
    python_path <- system.file("python/sklearn/cla_mlp.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]
  
  prediction <- mlp_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
