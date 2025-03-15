#' Synthetic Data
#' 
#' Minority class augmentation
#' @import reticulate

#' @description Generates synthetic samples
#' @param sampling_strategy Strategy parameter. Default is 'auto'
#' @param random_state Seed value
#' @return Python object
#' @export
create_smote_model <- function(sampling_strategy='auto', random_state=42) {
  reticulate::source_python("daltoolboxdp/inst/python/sklearn/imbalanced/smote.py")
  smote <- create_smote_model(sampling_strategy=sampling_strategy, random_state=random_state)
  return(smote)
}

#' Fit and resample dataset using SMOTE
#' @param model The SMOTE model
#' @param df_train Data frame to resample
#' @param target_column Target column name
#' @return List containing resampled features and target
#' @export
fit_resample <- function(model, df_train, target_column) {
  # Convert df_train to pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)
  
  # Separate features and target
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  
  # Perform resampling
  result <- model$fit_resample(X, y)
  
  # Convert back to R objects
  X_resampled <- reticulate::py_to_r(result[[1]])
  y_resampled <- reticulate::py_to_r(result[[2]])
  
  return(list(X_resampled, y_resampled))
}