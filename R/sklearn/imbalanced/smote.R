#'@title Imbalanced Data Handling with SMOTE and RandomForest
#'@description This module applies SMOTE oversampling followed by a RandomForest classification.
#'@import reticulate

#' Create model for oversampling
#'@return A Python SMOTE object
#'@export
create_smote_model <- function() {
  if (!exists("inbalanced_create_model")) {
    reticulate::source_python("path_to_your_python_script.py")
  }
  smote <- inbalanced_create_model()
  return(smote)
}

#' Fit and resample the dataset using oversampling
#'@param select_method The oversampling technique (Python object)
#'@param df_train Data frame to resample
#'@param target_column The target column name as string
#'@return Tuple of resampled X and y
#'@export
fit_resample <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  
  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  # Calling Python function for resampling
  list_resample <- select_method$fit_resample(df_train_py, target_column)
  
  # Convert the result back to R data frames/matrices
  X_train_smote <- reticulate::py_to_r(list_resample[[1]])
  y_train_smote <- reticulate::py_to_r(list_resample[[2]])
  
  return(list(X_train_smote, y_train_smote))