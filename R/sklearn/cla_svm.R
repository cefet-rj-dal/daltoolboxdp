#'@title Support Vector Classifier
#'@description Classifies using the SVC algorithm.
#' It wraps the sklearn library.
#'@param attribute attribute target to model building.
#'@param slevels Possible values for the target classification.
#'@param C Regularization parameter.
#'@param kernel Specifies the kernel type to be used in the algorithm.
#'@param degree Degree of the polynomial kernel function ('poly').
#'@param gamma Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
#'@param coef0 Independent term in kernel function.
#'@param probability Whether to enable probability estimates.
#'@param tol Tolerance for stopping criterion.
#'@param cache_size Size of the kernel cache (in MB).
#'@param class_weight Set the parameter class_weight.
#'@param verbose Enable verbose output.
#'@param max_iter Limit on iterations.
#'@param decision_function_shape One-vs-rest ('ovr') or one-vs-one ('ovo').
#'@param random_state Seed for random number generation.
#'@return A svc object.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_svc("Species", slevels, C=1.0, kernel="rbf")
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
cla_svc <- function(attribute, slevels,
                    C = 1.0,
                    kernel = "rbf",
                    degree = 3,
                    gamma = "scale",
                    coef0 = 0.0,
                    probability = FALSE,
                    tol = 0.001,
                    cache_size = 200,
                    class_weight = NULL,
                    verbose = FALSE,
                    max_iter = -1,
                    decision_function_shape = "ovr",
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
    tol = as.numeric(tol),
    cache_size = as.numeric(cache_size),
    class_weight = class_weight,
    verbose = verbose,
    max_iter = as.integer(max_iter),
    decision_function_shape = decision_function_shape,
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
      obj$tol,
      obj$cache_size,
      obj$class_weight,
      obj$verbose,
      obj$max_iter,
      obj$decision_function_shape,
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
