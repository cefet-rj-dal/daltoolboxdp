#' Tree Ensemble
#'@title Random Forest Classifier
#'@description Implements classification using Random Forest algorithm.
#' This function wraps the RandomForestClassifier from Python's scikit-learn library.
#'@param attribute Target attribute name for model building
#'@param slevels List of possible values for classification target
#'@param n_estimators Number of trees in random forest
#'@param criterion Function name for measuring split quality
#'@param max_depth Maximum tree depth value
#'@param min_samples_split Minimum samples needed for internal node split
#'@param min_samples_leaf Minimum samples needed at leaf node
#'@param min_weight_fraction_leaf Minimum weighted fraction value
#'@param max_features Number of features to consider for best split
#'@param max_leaf_nodes Maximum number of leaf nodes
#'@param min_impurity_decrease Minimum impurity decrease needed for split
#'@param bootstrap Whether to use bootstrap samples
#'@param oob_score Whether to use out-of-bag samples
#'@param n_jobs Number of parallel jobs
#'@param random_state Seed for random number generation
#'@param verbose Whether to enable verbose output
#'@param warm_start Whether to reuse previous solution
#'@param class_weight Weights associated with classes
#'@param ccp_alpha Complexity parameter value for pruning
#'@param max_samples Number of samples for training estimators
#'@param monotonic_cst Monotonicity constraints for features
#'@return A Random Forest classifier object
#'@return `cla_rf` object
#'@examples
#'#See an example of using `cla_rf` at this
#'#https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/cla_rf.md
#'@import daltoolbox
#'@export
cla_rf <- function(attribute, slevels, n_estimators=100, criterion='gini', max_depth=NULL, min_samples_split=2,
                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=NULL,
                   min_impurity_decrease=0.0, bootstrap=TRUE, oob_score=FALSE, n_jobs=NULL, random_state=NULL,
                   verbose=0, warm_start=FALSE, class_weight=NULL, ccp_alpha=0.0, max_samples=NULL,
                   monotonic_cst=NULL) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    n_estimators = as.integer(n_estimators), 
    criterion = criterion,
    max_depth = max_depth, 
    min_samples_split = as.integer(min_samples_split), 
    min_samples_leaf = as.integer(min_samples_leaf),
    min_weight_fraction_leaf = as.numeric(min_weight_fraction_leaf), 
    max_features = max_features,
    max_leaf_nodes = max_leaf_nodes, 
    min_impurity_decrease = as.numeric(min_impurity_decrease),
    bootstrap = bootstrap, 
    oob_score = oob_score, 
    n_jobs = n_jobs, 
    random_state = if(!is.null(random_state)) as.integer(random_state) else NULL,
    verbose = as.integer(verbose), 
    warm_start = warm_start, 
    class_weight = class_weight, 
    ccp_alpha = as.numeric(ccp_alpha),
    max_samples = max_samples, 
    monotonic_cst = monotonic_cst
  )
  
  obj <- c(obj, objex)
  class(obj) <- c("cla_rf", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit cla_rf
fit.cla_rf <- function(obj, data, ...) {
  python_path <- system.file("python/sklearn/cla_rf.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)
  
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

#'@import daltoolbox
#'@import reticulate
#'@export
predict.cla_rf  <- function(object, x, ...) {
  if (!exists("cla_rf_predict")) {
    python_path <- system.file("python/sklearn/cla_rf.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]
  
  prediction <- cla_rf_predict(object$model, x)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
