#'@title Multi-Layer Perceptron Classifier
#'@description Classifies using the MLP Classifier algorithm.
#' It wraps the sklearn library.
#'@param attribute attribute target to model building.
#'@param slevels Possible values for the target classification.
#'@param hidden_layer_sizes The ith element represents the number of neurons in the ith hidden layer.
#'@param activation Activation function for the hidden layer ('identity', 'logistic', 'tanh', 'relu').
#'@param solver The solver for weight optimization ('lbfgs', 'sgd', 'adam').
#'@param alpha L2 penalty (regularization term) parameter.
#'@param batch_size Size of minibatches for stochastic optimizers.
#'@param learning_rate Schedule for weight updates ('constant', 'invscaling', 'adaptive').
#'@param max_iter Maximum number of iterations.
#'@param tol Tolerance for optimization termination.
#'@param random_state Seed for random number generation.
#'@return A mlp object.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_mlp("Species", slevels, hidden_layer_sizes = c(100), max_iter = 300)
#'
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, iris)
#'train <- sr$train
#'test <- sr$test
#'
#'model <- fit(model, train)
#'
#'prediction <- predict(model, test)
#'predictand <- adjust_class_label(test[,"Species"])
#'test_eval <- evaluate(model, predictand, prediction)
#'test_eval$metrics
#'@export
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

#'@import reticulate
#'@export
fit.cla_mlp <- function(obj, data, ...) {
  # Source the Python file only if the function does not already exist
  if (!exists("mlp_create")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_mlp.py")
  }

  # Check if the model is already initialized, otherwise create it
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

  # Adjust the data frame if needed
  data <- adjust_data.frame(data)

  # Fit the model using the MLP Classifier function and the attributes from obj
  obj$model <- mlp_fit(obj$model, data, obj$attribute)

  return(obj)
}

#'@import reticulate
#'@export
predict.cla_mlp <- function(obj, data, ...) {
  if (!exists("mlp_predict"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_mlp.py")

  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]

  prediction <- mlp_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)

  return(prediction)
}
