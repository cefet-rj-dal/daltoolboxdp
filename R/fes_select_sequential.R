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

#' @rdname create_fe_sequential_fs_model
#' @describeIn create_fe_sequential_fs_model Fit and transform the dataset using SequentialFeatureSelector
#' @param select_method The SequentialFeatureSelector model (Python object)
#' @param df_train Data frame to transform
#' @param target_column The target column name as string
#' @return Transformed X_train with selected features
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

