#' Naive Bayes Classifier
#' 
#' @description Implements a classifier using the Gaussian Naive Bayes algorithm.
#' This function wraps the GaussianNB from Python's scikit-learn library.
#' @param attribute Target attribute name for model building (required)
#' @param slevels Possible values for the target classification (required)
#' @param var_smoothing Portion of the largest variance of all features that is added to variances
#' @param priors Prior probabilities of the classes. If specified must be a list of length n_classes
#' @return A Naive Bayes classifier object
#' @export
cla_nb <- function(attribute, slevels, var_smoothing=1e-9, priors=NULL) {
  obj <- list(
    attribute = attribute,
    slevels = slevels,
    var_smoothing = as.numeric(var_smoothing),
    priors = priors
  )
  
  class(obj) <- c("cla_nb", class(obj))
  return(obj)
}

#'@import reticulate
#'@export
fit.cla_nb <- function(obj, data, ...) {
  if (!exists("nb_create")) {
    python_path <- system.file("python/sklearn/cla_nb.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  if (is.null(obj$model)) {
    obj$model <- nb_create(
      var_smoothing = obj$var_smoothing,
      priors = obj$priors
    )
  }
  
  data <- adjust_data.frame(data)
  obj$model <- nb_fit(obj$model, data, obj$attribute)
  
  return(obj)
}

#'@import reticulate
#'@export
predict.cla_nb <- function(obj, data, ...) {
  if (!exists("nb_predict")) {
    python_path <- system.file("python/sklearn/cla_nb.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]
  
  prediction <- nb_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
