#' SVM Classifier
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
#'@return An SVM classifier object
#'@return `cla_svc` object
#'@examples
#'#See an example of using `cla_svc` at this
#'#https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/cla_svm.md
#'@import daltoolbox
#'@export
cla_svc <- function(attribute, slevels,
                    kernel = 'rbf',
                    degree = 3,
                    gamma = 'scale',
                    coef0 = 0.0,
                    tol = 0.001,
                    C = 1.0,
                    shrinking = TRUE,
                    probability = FALSE,
                    cache_size = 200,
                    class_weight = NULL,
                    verbose = FALSE,
                    max_iter = -1,
                    decision_function_shape = 'ovr',
                    break_ties = FALSE,
                    random_state = NULL) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    attribute = attribute,
    slevels = slevels,
    kernel = kernel,
    degree = as.integer(degree),
    gamma = gamma,
    coef0 = as.numeric(coef0),
    tol = as.numeric(tol),
    C = as.numeric(C),
    shrinking = shrinking,
    probability = probability,
    cache_size = as.numeric(cache_size),
    class_weight = class_weight,
    verbose = verbose,
    max_iter = as.integer(max_iter),
    decision_function_shape = decision_function_shape,
    break_ties = break_ties,
    random_state = if(!is.null(random_state)) as.integer(random_state) else NULL
  )
  
  obj <- c(obj, objex)
  class(obj) <- c("cla_svc", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit cla_svc
fit.cla_svc <- function(obj, data, ...) {
  python_path <- system.file("python/sklearn/cla_svc.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
  reticulate::source_python(python_path)
  
  if (is.null(obj$model)) {
    obj$model <- svc_create(
      obj$kernel,
      obj$degree,
      obj$gamma,
      obj$coef0,
      obj$tol,
      obj$C,
      obj$shrinking,
      obj$probability,
      obj$cache_size,
      obj$class_weight,
      obj$verbose,
      obj$max_iter,
      obj$decision_function_shape,
      obj$break_ties,
      obj$random_state
    )
  }
  
  data <- adjust_data.frame(data)
  obj$model <- svc_fit(obj$model, data, obj$attribute, obj$slevels)
  
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@export
predict.cla_svc <- function(object, x, ...) {
  if (!exists("svc_predict")) {
    python_path <- system.file("python/sklearn/cla_svc.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]
  
  prediction <- svc_predict(object$model, x)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
