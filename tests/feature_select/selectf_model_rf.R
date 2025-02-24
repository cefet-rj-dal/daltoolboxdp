# Load necessary R libraries
library(reticulate)
source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")
load_library("daltoolbox")
source("daltoolbox/R/sklearn/feature_select/selectf_model_rf.R")
# Load dataset
iris <- datasets::iris
head(iris)

# Encode the target variable
iris$species_encoded <- as.integer(as.factor(iris$Species))

# Split data into training and testing sets
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Prepare training data without target column
iris_train_label <- iris_train[, !names(iris_train) %in% c("Species")]

# Call Python RandomForest model and feature selection
rf_model <- create_fit_rf_model(iris_train_label, "species_encoded", n_estimators=100, random_state=0)
select_model <- create_fs_model(rf_model, threshold="mean", prefit=TRUE)

# Apply feature selection on training data
X_train_selected <- fit_transform_fs(select_model, iris_train_label, "species_encoded")

cat("Original shape:", dim(iris_train_label), "\n")  # Should be (150, 5)

# Print shape after feature selection
cat("Shape after RFE:", dim(X_train_selected), "\n")  # Should be (150, 2)

cat("Selected features:\n")
print(head(X_train_selected, 5))

