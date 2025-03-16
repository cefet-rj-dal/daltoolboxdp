#####################################################################
# Logistic Regression Feature Selection Testing Script
# Demonstrates feature selection using L1-regularized logistic regression
#####################################################################

# Import dependencies
# Load necessary R libraries
source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")
load_library("daltoolboxdp")

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
# Feature Selection Implementation
#--------------------
# Initialize logistic regression model with L1 regularization
lg_model <- create_fit_lg_model(iris_train_label, "species_encoded", 
                               C=0.1, penalty='l1', solver='liblinear')
select_model <- create_fs_model(lg_model, threshold="mean", prefit=TRUE)

# Apply LR-based feature selection
X_train_selected <- fit_transform_fs(select_model, iris_train_label, "species_encoded")

#--------------------
# Results Analysis
#--------------------
# Display dimensions and selected features
cat("Original shape:", dim(iris_train_label), "\n")  # Should be (150, 5)

# Print shape after feature selection
cat("Shape after RFE:", dim(X_train_selected), "\n")  # Should be (150, 2)

cat("Selected features:\n")
print(head(X_train_selected, 5))
