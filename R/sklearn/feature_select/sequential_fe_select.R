#'@title Feature Selection Using SequentialFeatureSelector
#'@description This module applies sequential feature selection using KNeighborsClassifier.
#'@import reticulate

#' Create a SequentialFeatureSelector model
#'@param n_neighbors The number of neighbors for KNN
#'@param direction The direction of feature selection, either "forward" or "backward"
#'@param n_features_to_select The number of features to select
#'@return A Python SequentialFeatureSelector object
#'@export
create_sequential_fs_model <- function(n_neighbors=3, direction="forward", n_features_to_select=2) {
  reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/sequential_fe_select.py") # Correct the file name if different

  sf_method <- fs_create(n_neighbors=n_neighbors, direction=direction, n_features_to_select=n_features_to_select)
  return(sf_method)
}

#' Fit and transform the dataset using SequentialFeatureSelector
#'@param select_method The SequentialFeatureSelector model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string
#'@return Transformed X_train with selected features
#'@export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_transform")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/sequential_fe_select.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  X_train_selected <- fit_transform(select_method, df_train_py, target_column)

  return(X_train_selected)
}

