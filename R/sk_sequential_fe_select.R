#' Sequential Feature Selection
#' 
#' Applies sequential feature selection using KNeighborsClassifier
#' @import reticulate

#' Create a sequential feature selector model
#' @param n_neighbors Number of neighbors for KNN (default: 3)
#' @param direction Direction of selection ("forward" or "backward")
#' @param n_features_to_select Number of features to select
#' @return A Python SequentialFeatureSelector object
#' @export
create_sequential_fs_model <- function(n_neighbors=3, 
                                     direction="forward", 
                                     n_features_to_select=2) {
  reticulate::source_python("inst/python/sklearn/feature_select/sequential_fe_select.py")
  model <- fs_create(n_neighbors=n_neighbors, 
                    direction=direction, 
                    n_features_to_select=n_features_to_select)
  return(model)
}

#' Fit and transform the dataset
#'@param model The SequentialFeatureSelector model
#'@param df_train Data frame to transform
#'@param target_column Target column name
#'@return Transformed features
#'@export
fit_transform <- function(model, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_transform")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/sequential_fe_select.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  result <- model$fit_transform(X, y)
  return(reticulate::py_to_r(result))
}

