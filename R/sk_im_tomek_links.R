#' Clean Samples
#' 
#' Remove samples from majority class
#' @import reticulate

#' Tomek Links Under-sampling
#' 
#' Wrapper functions for Tomek Links under-sampling method
#' @import reticulate

#' Create under-sampling model using Tomek Links method
#' @param sampling_strategy The sampling strategy to use. Default is 'auto'
#' @return A Python TomekLinks object
#' @export
create_tomek_model <- function(sampling_strategy='auto') {
  reticulate::source_python("inst/python/sklearn/imbalanced/tomek_links.py")
  tomek <- create_tomek_model(sampling_strategy=sampling_strategy)
  return(tomek)
}

#' Fit and resample dataset using Tomek Links
#' @param model The TomekLinks model
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
