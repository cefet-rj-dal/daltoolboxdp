#'@title Feature Selection Using SelectKBest
#'@description This module applies feature selection using SelectKBest with a specified score function.
#'@import reticulate

#' Create a SelectKBest feature selection model
#'@param k The number of top features to select
#'@return A Python SelectKBest object
#'@export
create_fs_model <- function(k=10) {
  reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectk_beast.py") # Correct the file name if different
  sf_method <- fs_create(k=k)
  return(sf_method)
}

#' Fit and transform the dataset using SelectKBest
#'@param select_method The SelectKBest model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string
#'@return Transformed X_train with selected features
#'@export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_transform")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectk_beast.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  X_train_selected <- fit_transform(select_method, df_train_py, target_column)

  return(X_train_selected)
}
