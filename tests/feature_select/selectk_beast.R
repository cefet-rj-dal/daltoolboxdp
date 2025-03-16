# Load necessary R libraries
source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")
load_library("daltoolboxdp")
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

select_model <- create_fs_model(k=10)

# Apply feature selection on training data
X_train_selected <- fit_transform_fs(select_model, iris_train_label, "species_encoded")

cat("Original shape:", dim(iris_train_label), "\n")  # Should be (150, 5)

cat("Shape after KBeast:", dim(X_train_selected), "\n")  # Should be (150, 2)

cat("Selected features:\n")
print(head(X_train_selected, 5))

