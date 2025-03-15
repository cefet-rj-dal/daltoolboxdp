#' Tree Boosting 
#' 
#' @description Ensemble learning by boosting trees
#' @import reticulate
#'@title Gradient Boosting Classifier
#'@description Implements a classifier using the Gradient Boosting algorithm.
#' This function wraps the GradientBoostingClassifier from Python's scikit-learn library.
#'@param attribute Target attribute name for model building
#'@param slevels Possible values for the target classification
#'@param loss Loss function to be optimized ('log_loss', 'exponential')
#'@param learning_rate Learning rate that shrinks the contribution of each tree
#'@param n_estimators Number of boosting stages to perform
#'@param subsample Fraction of samples to be used for fitting the individual base learners
#'@param criterion Function to measure the quality of a split
#'@param min_samples_split Minimum number of samples required to split an internal node
#'@param min_samples_leaf Minimum number of samples required to be at a leaf node
#'@param min_weight_fraction_leaf Minimum weighted fraction of the sum total of weights
#'@param max_depth Maximum depth of the individual regression estimators
#'@param min_impurity_decrease Minimum impurity decrease required for split
#'@param init Estimator object to initialize the model
#'@param random_state Random number generator seed
#'@param max_features Number of features to consider for best split
#'@param verbose Controls verbosity of the output
#'@param max_leaf_nodes Maximum number of leaf nodes
#'@param warm_start Whether to reuse solution of previous call
#'@param validation_fraction Proportion of training data to set aside for validation
#'@param n_iter_no_change Used to decide if early stopping will be used
#'@param tol Tolerance for early stopping
#'@param ccp_alpha Complexity parameter for cost-complexity pruning
#'@return A Gradient Boosting classifier object
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_gb("Species", slevels)
#'
#'# Preparing dataset for random sampling
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
cla_gb <- function(attribute, slevels,
                   loss = 'log_loss',
                   learning_rate = 0.1,
                   n_estimators = 100,
                   subsample = 1.0,
                   criterion = 'friedman_mse',
                   min_samples_split = 2,
                   min_samples_leaf = 1,
                   min_weight_fraction_leaf = 0.0,
                   max_depth = 3,
                   min_impurity_decrease = 0.0,
                   init = NULL,
                   random_state = NULL,
                   max_features = NULL,
                   verbose = 0,
                   max_leaf_nodes = NULL,
                   warm_start = FALSE,
                   validation_fraction = 0.1,
                   n_iter_no_change = NULL,
                   tol = 0.0001,
                   ccp_alpha = 0.0) {
  obj <- list(
    attribute = attribute,
    slevels = slevels,
    loss = loss,
    learning_rate = as.numeric(learning_rate),
    n_estimators = as.integer(n_estimators),
    subsample = as.numeric(subsample),
    criterion = criterion,
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    min_weight_fraction_leaf = as.numeric(min_weight_fraction_leaf),
    max_depth = as.integer(max_depth),
    min_impurity_decrease = as.numeric(min_impurity_decrease),
    init = init,
    random_state = if(!is.null(random_state)) as.integer(random_state) else NULL,
    max_features = max_features,
    verbose = as.integer(verbose),
    max_leaf_nodes = if(!is.null(max_leaf_nodes)) as.integer(max_leaf_nodes) else NULL,
    warm_start = warm_start,
    validation_fraction = as.numeric(validation_fraction),
    n_iter_no_change = if(!is.null(n_iter_no_change)) as.integer(n_iter_no_change) else NULL,
    tol = as.numeric(tol),
    ccp_alpha = as.numeric(ccp_alpha)
  )

  class(obj) <- c("cla_gb", class(obj))
  return(obj)
}

#'@import reticulate
#'@export
fit.cla_gb <- function(obj, data, ...) {
  # Source the Python file only if the function does not already exist
  if (!exists("cla_gb_create")) {
    reticulate::source_python("daltoolboxdp/inst/python/sklearn/cla_gb.py")
  }

  # Check if the model is already initialized, otherwise create it
  if (is.null(obj$model)) {
    obj$model <- cla_gb_create(
      obj$loss,
      obj$learning_rate,
      obj$n_estimators,
      obj$subsample,
      obj$criterion,
      obj$min_samples_split,
      obj$min_samples_leaf,
      obj$min_weight_fraction_leaf,
      obj$max_depth,
      obj$min_impurity_decrease,
      obj$init,
      obj$random_state,
      obj$max_features,
      obj$verbose,
      obj$max_leaf_nodes,
      obj$warm_start,
      obj$validation_fraction,
      obj$n_iter_no_change,
      obj$tol,
      obj$ccp_alpha
    )
  }

  # Adjust the data frame if needed
  data <- adjust_data.frame(data)

  # Fit the model using the Gradient Boosting function and the attributes from obj
  obj$model <- cla_gb_fit(obj$model, data, obj$attribute, obj$slevels)

  return(obj)
}

#'@import reticulate
#'@export
predict.cla_gb  <- function(obj, data, ...) {
  if (!exists("cla_gb_predict"))
    reticulate::source_python("inst/python/sklearn/cla_gb.py")

  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]

  prediction <- cla_gb_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)

  return(prediction)
}
