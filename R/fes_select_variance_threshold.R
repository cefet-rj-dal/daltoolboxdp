#' @title Feature Selection Using VarianceThreshold
#' @description Applies feature selection based on variance using scikit-learn's VarianceThreshold.
#' @param threshold The variance threshold below which features will be removed
#' @return A Python VarianceThreshold object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_fe_variance_threshold_model <- function(threshold=0.2) {
  python_path <- system.file(
    "python/feature_select/variance_threshold.py",
    package = "daltoolboxdp"
  )
  reticulate::source_python(python_path)
  vt_model <- fs_create(threshold = threshold)
  return(vt_model)
}

#' @rdname create_fe_variance_threshold_model
#' @param select_method The VarianceThreshold model (Python object)
#' @param df_train Data frame to transform
#' @param target_column The target column name as string (unused for VarianceThreshold)
#' @return Transformed feature matrix or data frame
#' @export
fit_transform_fe_variance_threshold <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file(
    "python/feature_select/variance_threshold.py",
    package = "daltoolboxdp"
  )
  reticulate::source_python(python_path)

  df_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_py)     # VarianceThreshold ignores target_column
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
