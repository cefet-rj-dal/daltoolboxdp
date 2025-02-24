#'@title Feature Selection Using VarianceThreshold
#'@description This module applies feature selection based on variance using VarianceThreshold.
#'@import reticulate

#' Create a VarianceThreshold model
#'@param threshold The variance threshold below which features will be removed
#'@return A Python VarianceThreshold object
#'@export
create_variance_threshold_model <- function(threshold=0.2) {
  reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/variance_threshold.py")  # Make sure the file name is correct
  sf_method <- fs_create(threshold=threshold)
  return(sf_method)
}

#' Fit and transform the dataset using VarianceThreshold
#'@param select_method The VarianceThreshold model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string (not used for fitting)
#'@return Transformed X_train with features above the variance threshold
#'@export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_transform")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/variance_threshold.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  X_train_selected <- fit_transform(select_method, df_train_py, target_column)

  return(X_train_selected)
}
