#'@title Feature Selection Using Smote
#'@description This module applies feature selection based on variance using Smote.
#'@import reticulate

#' Create a Smote model
#'@param random_state
#'@return A Python Smote object
#'@export
create_smote_model <- function(random_state=42) {
  reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/smote.py")  # Make sure the file name is correct
  sf_method <- inbalanced_create_model(random_state=random_state)
  return(sf_method)
}

#' Fit and transform the dataset using Smote
#'@param select_method The Smote model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string (not used for fitting)
#'@return Transformed X_train with features above the variance threshold
#'@export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_resample")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/smote.py")
  }
  list_resample <- fit_resample(select_method, df_train, target_column)

  # Convert the results back to R data frames/matrices
  X_train_resampled <- list_resample[[1]]
  y_train_resampled <- list_resample[[2]]

  return(list(X_train_resampled, y_train_resampled))
}
