#' @title Feature Selection Using SequentialFeatureSelector
#' @description This module applies sequential feature selection using KNeighborsClassifier.
#' @param n_neighbors Number of neighbors for KNN
#' @param direction Direction of selection: "forward" or "backward"
#' @param n_features_to_select Number of features to select
#' @return A Python SequentialFeatureSelector object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_fe_sequential_fs_model <- function(n_neighbors=3, direction="forward", n_features_to_select=2) {
  python_path <- system.file("python/feature_select/sequential_fe_select.py", package="daltoolboxdp")
  reticulate::source_python(python_path)

  sf_method <- fs_create(n_neighbors=n_neighbors, direction=direction, n_features_to_select=n_features_to_select)
  return(sf_method)
}

#' @title Fit and Transform with Sequential Feature Selector
#' @description Apply a pre‐configured SequentialFeatureSelector (Python) on a data frame and return only the selected features.
#' @param select_method A SequentialFeatureSelector Python object created by `create_fe_sequential_fs_model()`.
#' @param df_train An R data frame of training data.
#' @param target_column Character name of the target column in `df_train`.
#' @return A data frame containing only the features selected by the SequentialFeatureSelector.
#' @export
fit_transform_fe_sequential_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/feature_select/sequential_fe_select.py", package="daltoolboxdp")
  reticulate::source_python(python_path)

  # Convert df_train to a pandas DataFrame
  df_py <- reticulate::r_to_py(df_train)

  X_py <- fit_transform(select_method, df_py, target_column)
  X_sel <- reticulate::py_to_r(X_py)

  return(X_sel)
}

