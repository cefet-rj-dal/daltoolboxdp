#'@title Feature Selection Using VarianceThreshold
#'@description This module applies feature selection based on variance using VarianceThreshold.
#'@import reticulate

#' Create a VarianceThreshold model
#'@param threshold The variance threshold below which features will be removed
#'@return A Python VarianceThreshold object
#'@export
create_variance_threshold_model <- function(threshold=0.2) {
  reticulate::source_python("inst/python/sklearn/feature_select/variance_threshold.py")
  model <- fs_create(threshold=threshold)
  return(model)
}

#' Fit and transform the dataset using VarianceThreshold
#'@param model The VarianceThreshold model
#'@param df_train Data frame to transform
#'@param target_column Target column name
#'@return Transformed features
#'@export
fit_transform <- function(model, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_transform")) {
    reticulate::source_python("inst/python/sklearn/feature_select/variance_threshold.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)
  X <- df_train_py$drop(target_column, axis=1)
  result <- model$fit_transform(X)
  return(reticulate::py_to_r(result))
}
