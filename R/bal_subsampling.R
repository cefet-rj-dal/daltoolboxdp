#'@title Subsampling
#'@description Subsampling balances class distributions by reducing the representation
#' of majority classes through random under-sampling.
#'
#'@param attribute Character. Name of the target class attribute to balance.
#'@return A `bal_subsampling` object.
#'
#'@references
#' Kubat, M., & Matwin, S. (1997). Addressing the Curse of Imbalanced Training Sets: One-Sided Selection.
#' Drummond, C., & Holte, R. (2003). C4.5, Class Imbalance, and Cost Sensitivity.
#'
#'@examples
#'\dontrun{
#'set.seed(123)
#'data(iris)
#'mod_iris <- iris[c(1:50, 51:71, 101:111), ]   # induce imbalance
#'table(mod_iris$Species)
#'
#'bal <- bal_subsampling('Species')              # random under-sampling
#'bal <- daltoolbox::fit(bal, mod_iris)
#'adjust_iris <- daltoolbox::transform(bal, mod_iris)
#'table(adjust_iris$Species)                     # all classes at minority count
#'}
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
bal_subsampling <- function(attribute) {
  obj <- dal_transform()
  obj$attribute <- attribute
  class(obj) <- append("bal_subsampling", class(obj))
  return(obj)
}

#'@importFrom daltoolbox transform
#'@export
transform.bal_subsampling <- function(obj, data, ...) {
  # Randomly downsample each class to match the minority count
  data <- data
  attribute <- obj$attribute
  x <- sort((table(data[,attribute])))
  qminor = as.integer(x[1])
  newdata = NULL
  for (i in 1:length(x)) {
    curdata = data[data[,attribute]==(names(x)[i]),]
    idx = sample(1:nrow(curdata),qminor)
    curdata = curdata[idx,]
    newdata = rbind(newdata, curdata)
  }
  data <- newdata
  return(data)
}
