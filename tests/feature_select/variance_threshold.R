#####################################################################
# Variance Threshold Feature Selection Testing Script
# Demonstrates feature selection based on variance threshold
#####################################################################

# Load necessary R libraries
library(reticulate)
source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")
load_library("daltoolbox")
source("daltoolbox/R/sklearn/feature_select/variance_threshold.R")

#--------------------
# Data Preprocessing
#--------------------
# Load and prepare iris dataset
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

#--------------------
# Feature Selection
#--------------------
# Initialize Variance Threshold selector with threshold=0.2
select_model <- create_variance_threshold_model(threshold=0.2)

# Apply variance-based feature selection
X_train_filtered <- fit_transform_fs(select_model, iris_train_label, "species_encoded")

#--------------------
# Results Analysis
#--------------------
# Display dimensions and selected features
cat("Original shape:", dim(iris_train_label), "\n")  # Should be (150, 5)

cat("Shape after Sequential VarianceThreshold:", dim(X_train_selected), "\n")  # Should be (150, 2)

cat("Selected features:\n")
print(head(X_train_selected, 5))
