source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")
load_library("daltoolbox")
source("daltoolbox/R/sklearn/cla_knn.R")


evaluate <- function(obj, data, prediction, ...)
{
  result <- list(data = data, prediction = prediction)
  adjust_predictions <- function(predictions) {
    predictions_i <- matrix(rep.int(0, nrow(predictions) *
                                      ncol(predictions)), nrow = nrow(predictions), ncol = ncol(predictions))
    y <- apply(predictions, 1, nnet::which.is.max)
    for (i in unique(y)) {
      predictions_i[y == i, i] <- 1
    }
    return(predictions_i)
  }
  predictions <- adjust_predictions(result$prediction)
  result$conf_mat <- MLmetrics::ConfusionMatrix(data, predictions)
  result$accuracy <- MLmetrics::Accuracy(y_pred = predictions,
                                         y_true = data)
  result$f1 <- MLmetrics::F1_Score(y_pred = predictions, y_true = data,
                                   positive = 1)
  result$sensitivity <- MLmetrics::Sensitivity(y_pred = predictions,
                                               y_true = data, positive = 1)
  result$specificity <- MLmetrics::Specificity(y_pred = predictions,
                                               y_true = data, positive = 1)
  result$precision <- MLmetrics::Precision(y_pred = predictions,
                                           y_true = data, positive = 1)
  result$recall <- MLmetrics::Recall(y_pred = predictions,
                                     y_true = data, positive = 1)
  result$metrics <- data.frame(accuracy = result$accuracy,
                               f1 = result$f1, sensitivity = result$sensitivity, specificity = result$specificity,
                               precision = result$precision, recall = result$recall)
  return(result)
}
iris <- datasets::iris
head(iris)

slevels <- levels(iris$Species)
slevels

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test


tbl <- rbind(table(iris[, "Species"]),
             table(iris_train[, "Species"]),
             table(iris_test[, "Species"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)

iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- cla_knn("species_encoded", slevels, n_neighbors=1)
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)

iris_train_predictand <- adjust_class_label(iris_train_label[, "species_encoded"])

colnames(train_prediction) <- c("setosa", "versicolor", "virginica")
colnames(iris_train_predictand) <- c("setosa", "versicolor", "virginica")

train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)


iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)


iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
