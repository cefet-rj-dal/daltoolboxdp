#'@title Relief
#'@description Relief ranks features by how well they differentiate between instances of
#' different classes in local neighborhoods. Wraps `FSelector`'s Relief.
#'
#'@param attribute Character. Name of the (categorical) target variable.
#'@return A `fs_relief` object.
#'
#'@references
#' Kira, K., & Rendell, L. A. (1992). The Feature Selection Problem: Traditional Methods and a New Algorithm.
#' Kononenko, I. (1994). Estimating attributes: analysis and extensions of Relief.
#'
#'@examples
#'data(iris)
#'fs <- daltoolbox::fit(fs_relief("Species"), iris)
#'data_relief <- daltoolbox::transform(fs, iris)
#'head(data_relief)
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
fs_relief <- function(attribute) {
  obj <- fs(attribute)
  class(obj) <- append("fs_relief", class(obj))
  return(obj)
}

#'@importFrom daltoolbox fit
#'@importFrom FSelector relief
#'@importFrom doBy orderBy
#'@importFrom stats coef formula predict
#'@export
fit.fs_relief <- function(obj, data, ...) {
  data <- data.frame(data)
  data[, obj$attribute] <- as.factor(data[, obj$attribute])

  class_formula <- stats::formula(paste(obj$attribute, "  ~ ."))
  weights <-FSelector::relief(class_formula, data)

  tab <- data.frame(weights)
  tab <- doBy::orderBy(~-attr_importance, data = tab)
  tab$i <- row(tab)
  tab$import_acum <- cumsum(tab$attr_importance)
  myfit <- daltoolbox::fit_curvature_min()
  res <- daltoolbox::transform(myfit, tab$import_acum)
  tab <- tab[tab$import_acum <= res$y, ]
  vec <- rownames(tab)

  obj$features <- vec

  return(obj)
}
