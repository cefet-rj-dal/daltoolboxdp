#'@title Inbalanced Data Handling with Tomek Links and RandomForest
#'@description This module applies Tomek Links under-sampling followed by a RandomForest classification.
#'@import reticulate


#' Create model for under-sampling
#'@return A Python TomekLinks object
#'@export
create_tomek_model <- function() {
  if (!exists("inbalanced_create_model")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/imbalanced/tomek_links.py")
  }
  tomek <- inbalanced_create_model()
  return(tomek)
}


#' Fit and resample the dataset using under-sampling
#'@param select_method The under-sampling technique (Python object)
#'@param df_train Data frame to resample
#'@param target_column The target column name as string
#'@return Tuple of resampled X and y
#'@export
fit_resample <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  X_train <- df_train[, !(names(df_train) %in% target_column), drop = FALSE]
  y_train <- df_train[[target_column]]

  # Calling Python function for resampling
  list_resample <- select_method$fit_resample(X_train, y_train)
  X_train_smote <- list_resample[[1]]
  y_train_smote <- list_resample[[2]]

  return(list(X_train_smote, y_train_smote))
}
