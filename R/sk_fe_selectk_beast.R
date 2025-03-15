#' K-Best Feature Selection
#' 
#' Feature selection using SelectKBest method
#' @import reticulate

#' Create k-best feature selector
#' @param k Number of top features to select
#' @return A Python SelectKBest object
#' @export
create_selectk_model <- function(k=10) {
  reticulate::source_python("inst/python/sklearn/feature_select/selectk_beast.py")
  model <- fs_create(k=k)
  return(model)
}

#' Fit and transform the dataset
#' @param model The SelectKBest model
#' @param df_train Data frame to transform
#' @param target_column Target column name
#' @return Transformed features
#' @export
fit_transform <- function(model, df_train, target_column) {
  df_train_py <- reticulate::r_to_py(df_train)
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  result <- model$fit_transform(X, y)
  return(reticulate::py_to_r(result))
}
