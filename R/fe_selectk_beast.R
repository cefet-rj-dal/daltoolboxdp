#' @title Feature Selection Using SelectKBest
#' @description Applies feature selection using SelectKBest from scikit-learn.
#' @param k Number of top features to select
#' @return A Python SelectKBest object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_fe_selectkbest_model <- function(k=10) {
  python_path <- system.file("python/sklearn/feature_select/selectk_beast.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  sf_method <- fs_create(k=k)
  return(sf_method)
}

#' @rdname create_fe_selectkbest_model
#' @describeIn create_fe_selectkbest_model Fit and transform the dataset using SelectKBest
#' @param select_method The SelectKBest model (Python object)
#' @param df_train Data frame to transform
#' @param target_column The target column name as string
#' @return Transformed X_train with selected features
#' @export
fit_transform_fe_selectkbest <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/sklearn/feature_select/selectk_beast.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_train_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_train_py, target_column)
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
