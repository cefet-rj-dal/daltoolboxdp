# #####################################################################
# # Tomek Links Testing Script
# # Demonstrates under-sampling using Tomek Links for imbalanced datasets
# #####################################################################

# # Import required libraries and source files
# # Load necessary R libraries
# library(reticulate)
# source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")
# load_library("daltoolbox")
# source("daltoolbox/R/sklearn/imbalanced/tomek_links.R")
# source("daltoolbox/R/sklearn/cla_rf.R")

# #--------------------
# # Evaluation Function
# #--------------------
# evaluate <- function(obj, data, prediction, ...)
# {
#   result <- list(data = data, prediction = prediction)
#   adjust_predictions <- function(predictions) {
#     predictions_i <- matrix(rep.int(0, nrow(predictions) *
#                                       ncol(predictions)), nrow = nrow(predictions), ncol = ncol(predictions))
#     y <- apply(predictions, 1, nnet::which.is.max)
#     for (i in unique(y)) {
#       predictions_i[y == i, i] <- 1
#     }
#     return(predictions_i)
#   }
#   predictions <- adjust_predictions(result$prediction)
#   result$conf_mat <- MLmetrics::ConfusionMatrix(data, predictions)
#   result$accuracy <- MLmetrics::Accuracy(y_pred = predictions,
#                                          y_true = data)
#   result$f1 <- MLmetrics::F1_Score(y_pred = predictions, y_true = data,
#                                    positive = 1)
#   result$sensitivity <- MLmetrics::Sensitivity(y_pred = predictions,
#                                                y_true = data, positive = 1)
#   result$specificity <- MLmetrics::Specificity(y_pred = predictions,
#                                                y_true = data, positive = 1)
#   result$precision <- MLmetrics::Precision(y_pred = predictions,
#                                            y_true = data, positive = 1)
#   result$recall <- MLmetrics::Recall(y_pred = predictions,
#                                      y_true = data, positive = 1)
#   result$metrics <- data.frame(accuracy = result$accuracy,
#                                f1 = result$f1, sensitivity = result$sensitivity, specificity = result$specificity,
#                                precision = result$precision, recall = result$recall)
#   return(result)
# }

# #--------------------
# # Data Preparation
# #--------------------
# # Load and prepare iris dataset
# # Load dataset
# iris <- datasets::iris
# head(iris)

# # Encode the target variable
# iris$species_encoded <- as.integer(as.factor(iris$Species))

# # Split data into training and testing sets
# slevels <- levels(iris$Species)

# set.seed(1)
# sr <- sample_random()
# sr <- train_test(sr, iris)
# iris_train <- sr$train
# iris_test <- sr$test

# # Prepare training data without target column
# iris_train_label <- iris_train[, !names(iris_train) %in% c("Species")]

# #--------------------
# # Tomek Links Implementation
# #--------------------
# # Initialize Tomek Links model for under-sampling
# select_model <- create_tomek_model()

# # Apply Tomek Links under-sampling
# list_tomek_model <- fit_resample(select_model, iris_train_label, "species_encoded")

# X_train_tomek <- list_tomek_model[[1]]
# y_train_tomek <- list_tomek_model[[2]]

# merged_df <- data.frame(X_train_tomek, species_encoded = y_train_tomek)

# #--------------------
# # Model Training and Evaluation
# #--------------------
# # Train Random Forest on balanced dataset
# rf_tomek <- cla_rf("species_encoded", slevels)
# rf_tomek <- fit(rf_tomek, merged_df)

# iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
# iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
# test_prediction <- predict(rf_tomek, iris_test_label)

# # Display class distribution after Tomek Links
# cat("Class distribution after Tomek Links under-sampling:\n")
# print(table(merged_df$species_encoded))

# iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
# test_eval <- evaluate(rf_tomek, iris_test_predictand, test_prediction)
# print(test_eval$metrics)
