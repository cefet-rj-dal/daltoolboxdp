#' Support Vector Machine Classification
#' 
#' Classification using SVM algorithm from scikit-learn
#' @import reticulate
#'@title Support Vector Machine Classification
#'@description Implements classification using Support Vector Machine (SVM) algorithm.
#' This function wraps the SVC classifier from Python's scikit-learn library.
#'@param attribute Target attribute name for model building
#'@param slevels List of possible values for classification target
#'@param C Regularization strength parameter
#'@param kernel Kernel function type ('linear', 'poly', 'rbf', 'sigmoid')
#'@param degree Polynomial degree when using 'poly' kernel
#'@param gamma Kernel coefficient value
#'@param coef0 Independent term value in kernel function
#'@param probability Whether to enable probability estimates
#'@param shrinking Whether to use shrinking heuristic
#'@param tol Tolerance value for stopping criterion
#'@param cache_size Kernel cache size value in MB
#'@param class_weight Weights associated with classes
#'@param verbose Whether to enable verbose output
#'@param max_iter Maximum number of iterations
#'@param decision_function_shape Shape of decision function ('ovo', 'ovr')
#'@param break_ties Whether to break tie decisions
#'@param random_state Seed for random number generation
#'@return An SVM classifier object configured for scikit-learn
#'@examples
#' # Load example dataset
#' data(iris)
#' slevels <- levels(iris$Species)
#' model <- cla_svc("Species", slevels, C=1.0, kernel="rbf")
#' 
#' # Initialize random sampling
#' sr <- sample_random()
#' sr <- train_test(sr, iris)
#' train <- sr$train
#' test <- sr$test
#' 
#' # Train and evaluate model
#' model <- fit(model, train)
#' prediction <- predict(model, test)
#' predictand <- adjust_class_label(test[,"Species"])
#' test_eval <- evaluate(model, predictand, prediction)
#' test_eval$metrics
#'@export
cla_svc <- function(attribute, slevels,
                    C = 1.0,
                    kernel = "rbf",
                    degree = 3,
                    gamma = "scale",
                    coef0 = 0.0,
                    probability = FALSE,
                    shrinking = TRUE,
                    tol = 0.001,
                    cache_size = 200,
                    class_weight = NULL,
                    verbose = FALSE,
                    max_iter = -1,
                    decision_function_shape = "ovr",
                    break_ties = FALSE,
                    random_state = NULL) {
  obj <- list(
    attribute = attribute,
    slevels = slevels,
    C = as.numeric(C),
    kernel = kernel,
    degree = as.integer(degree),
    gamma = gamma,
    coef0 = as.numeric(coef0),
    probability = probability,
    shrinking = shrinking,
    tol = as.numeric(tol),
    cache_size = as.numeric(cache_size),
    class_weight = class_weight,
    verbose = verbose,
    max_iter = as.integer(max_iter),
    decision_function_shape = decision_function_shape,
    break_ties = break_ties,
    random_state = if(!is.null(random_state)) as.integer(random_state) else NULL
  )

  class(obj) <- c("cla_svc", class(obj))
  return(obj)
}

#'@import reticulate
#'@export
fit.cla_svc <- function(obj, data, ...) {
  # Source the Python file only if the function does not already exist
  if (!exists("svc_create")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_svc.py")
  }

  # Check if the model is already initialized, otherwise create it
  if (is.null(obj$model)) {
    obj$model <- svc_create(
      obj$C,
      obj$kernel,
      obj$degree,
      obj$gamma,
      obj$coef0,
      obj$probability,
      obj$shrinking,
      obj$tol,
      obj$cache_size,
      obj$class_weight,
      obj$verbose,
      obj$max_iter,
      obj$decision_function_shape,
      obj$break_ties,
      obj$random_state
    )
  }

  # Adjust the data frame if needed
  data <- adjust_data.frame(data)

  # Fit the model using the SVC function and the attributes from obj
  obj$model <- svc_fit(obj$model, data, obj$attribute, obj$slevels)

  return(obj)
}

#'@import reticulate
#'@export
predict.cla_svc  <- function(obj, data, ...) {
  if (!exists("svc_predict"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_svc.py")

  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]

  prediction <- svc_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)

  return(prediction)
}
