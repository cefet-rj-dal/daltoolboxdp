#'@title Feature Selection
#'@description Base constructor for feature selection workflows. It stores the target attribute
#' and provides a simple transform that filters columns to the selected set.
#'
#'@details Concrete strategies such as information gain, Relief, LASSO, and forward
#' stepwise selection are available via `fs_ig()`, `fs_relief()`, `fs_lasso()`, and `fs_fss()`.
#'
#'@param attribute Character. Name of the target variable (predictand).
#'@return A `fs` object used as a base for feature selection strategies.
#'
#'@examples
#'\dontrun{
#'# Typical usage pattern:
#'# 1) Choose a strategy (e.g., fs_ig for information gain)
#'data(iris)
#'fs_sel <- daltoolbox::fit(fs_ig("Species"), iris)
#'fs_sel$features                 # selected feature names
#'
#'# 2) Apply selection to keep only chosen features + target
#'iris_small <- daltoolbox::transform(fs_sel, iris)
#'names(iris_small)
#'}
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
fs <- function(attribute) {
  obj <- daltoolbox::dal_transform()
  obj$attribute <- attribute
  class(obj) <- append("fs", class(obj))
  return(obj)
}

#'@importFrom daltoolbox transform
#'@export
transform.fs <- function(obj, data, ...) {
  # Keep only the selected features and the target attribute
  data <- data[, c(obj$features, obj$attribute)]
  return(data)
}


