#'@title Random Forest Classifier
#'@description Classifies using the Random Forest Classifier algorithm.
#' It wraps the sklearn library.
#'@param attribute attribute target to model building.
#'@param slevels Possible values for the target classification.
#'@param k A vector of integers indicating the number of neighbors to be considered.
#'@return A rf object.
#'@examples



#' @export
cla_rf <- function(attribute, slevels, n_estimators=100, criterion='gini', max_depth=NULL, min_samples_split=2,
                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=NULL,
                   min_impurity_decrease=0.0, bootstrap=TRUE, oob_score=FALSE, n_jobs=NULL, random_state=NULL,
                   verbose=0, warm_start=FALSE, class_weight=NULL, ccp_alpha=0.0, max_samples=NULL,
                   monotonic_cst=NULL) {
  obj <- list(attribute = attribute, slevels = slevels, n_estimators = n_estimators, criterion = criterion,
              max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf,
              min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features,
              max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease,
              bootstrap = bootstrap, oob_score = oob_score, n_jobs = n_jobs, random_state = random_state,
              verbose = verbose, warm_start = warm_start, class_weight = class_weight, ccp_alpha = ccp_alpha,
              max_samples = max_samples, monotonic_cst = monotonic_cst)

  class(obj) <- c("cla_rf", class(obj))
  return(obj)
}

#' @import reticulate
#' @export
fit.cla_rf <- function(obj, data, ...) {
  # Source the Python file only if the function does not already exist
  if (!exists("cla_rf_create"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_rf.py")

  if (is.null(obj$model)) {
    obj$model <- cla_rf_create(
      as.integer(obj$n_estimators),
      obj$criterion,
      obj$max_depth,
      as.integer(obj$min_samples_split),
      as.integer(obj$min_samples_leaf),
      obj$min_weight_fraction_leaf,
      obj$max_features,
      obj$max_leaf_nodes,
      obj$min_impurity_decrease,
      obj$bootstrap,
      obj$oob_score,
      obj$n_jobs,
      obj$random_state,
      as.integer(obj$verbose),
      obj$warm_start,
      obj$class_weight,
      obj$ccp_alpha,
      obj$max_samples,
      obj$monotonic_cst
    )
  }

  # Adjust the data frame
  data <- adjust_data.frame(data)

  obj$model <- cla_rf_fit(obj$model, data, obj$attribute)

  return(obj)
}


#'@import reticulate
#'@export
predict.cla_rf  <- function(obj, data, ...) {
  if (!exists("cla_rf_predict"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_rf.py")

  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]

  prediction <- cla_rf_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)

  return(prediction)
}
