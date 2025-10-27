#'@title Information Gain
#'@description Information Gain (IG) is an information-theoretic feature selection technique
#' that measures reduction in entropy of the target when a feature is observed. Wraps the
#' `FSelector` package.
#'
#'@param attribute Character. Name of the target variable.
#'@return A `fs_ig` object.
#'
#'@references
#' Quinlan, J. R. (1986). Induction of Decision Trees.
#'
#'@examples
#'\dontrun{
#'data(iris)
#'
#'# 1) Ensure target is a factor for IG-based ranking
#'iris2 <- iris
#'iris2$Species <- as.factor(iris2$Species)
#'
#'# 2) Fit selector and inspect chosen features
#'fs <- daltoolbox::fit(fs_ig("Species"), iris2)
#'fs$features                     # names of selected predictors
#'
#'# 3) Subset data to selected features + target
#'data_ig <- daltoolbox::transform(fs, iris2)
#'head(data_ig)
#'}
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
fs_ig <- function(attribute) {
  obj <- fs(attribute)
  class(obj) <- append("fs_ig", class(obj))
  return(obj)
}

#'@importFrom FSelector information.gain
#'@importFrom doBy orderBy
#'@importFrom daltoolbox fit
#'@export
fit.fs_ig <- function(obj, data, ...) {
  data <- data.frame(data)
  data[,obj$attribute] = as.factor(data[, obj$attribute])

  class_formula <- formula(paste(obj$attribute, "  ~ ."))
  weights <- FSelector::information.gain(class_formula, data)

  tab <- data.frame(weights)
  tab <- doBy::orderBy(~-attr_importance, data=tab)
  tab$i <- row(tab)
  tab$import_acum <- cumsum(tab$attr_importance)
  myfit <- daltoolbox::fit_curvature_min()
  res <- daltoolbox::transform(myfit, tab$import_acum)
  tab <- tab[tab$import_acum <= res$y, ]
  vec <- rownames(tab)

  obj$features <- vec

  return(obj)
}

