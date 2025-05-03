#' Naive Bayes Classifier
#'@title Gaussian Naive Bayes Classifier
#'@description Implements classification using the Gaussian Naive Bayes algorithm.
#' This function wraps the GaussianNB from Python's scikit-learn library.
#'@param attribute Target attribute name for model building
#'@param slevels List of possible values for classification target
#'@param var_smoothing Portion of the largest variance of all features that is added to variances
#'@param priors Prior probabilities of the classes. If specified must be a list of length n_classes
#'@return A Naive Bayes classifier object
#'@return `skcla_nb` object
#'@examples
#'#See an example of using `skcla_nb` at this
#'#https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/skcla_nb.md
#'@import daltoolbox
#'@export
skcla_nb <- function(attribute, slevels, var_smoothing=1e-9, priors=NULL) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    var_smoothing = as.numeric(var_smoothing),
    priors = priors
  )
  
  obj <- c(obj, objex)
  class(obj) <- c("skcla_nb", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit skcla_nb
fit.skcla_nb <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_nb.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)
  
  if (any(is.na(data))) {
    warning("Missing values detected in the data. These will be handled as part of the process.")
  }
  
  if (is.null(obj$model)) {
    obj$model <- skcla_nb_create(
      priors = obj$priors,
      var_smoothing = obj$var_smoothing
    )
    
    if (is.null(obj$model)) {
      stop("Failed to create Naive Bayes model.")
    }
  }
  
  data <- adjust_data.frame(data)
  
  if (!obj$attribute %in% names(data)) {
    stop(paste("Attribute", obj$attribute, "not found in the data."))
  }
  
  #message("Fitting model with data dimensions: ", nrow(data), " x ", ncol(data))
  #message("Target attribute: ", obj$attribute)
  
  obj$model <- skcla_nb_fit(obj$model, data, obj$attribute)
  
  if (is.null(obj$model)) {
    stop("Failed to fit Naive Bayes model.")
  }
  
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@export
predict.skcla_nb <- function(object, x, ...) {
  if (!exists("skcla_nb_predict")) {
    python_path <- system.file("python/skcla_nb.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  if (any(is.na(x))) {
    warning("Missing values detected in the prediction data. These will be handled as part of the process.")
  }
  
  x <- adjust_data.frame(x)
  
  if (object$attribute %in% names(x)) {
    x <- x[, !names(x) %in% object$attribute]
  }
  
  #message("Predicting with data dimensions: ", nrow(x), " x ", ncol(x))
  
  prediction <- skcla_nb_predict(object$model, x)
  
  if (is.null(prediction) || length(prediction) == 0) {
    warning("Prediction returned NULL or empty. Returning NA values.")
    prediction <- rep(NA, nrow(x))
  }
  
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
