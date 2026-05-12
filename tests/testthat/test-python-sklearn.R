make_skcla_data <- function() {
  data.frame(
    x1 = c(0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.05, 0.95),
    x2 = c(0.2, 0.1, 0.9, 0.8, 0.25, 0.75, 0.05, 0.95),
    class = factor(c("a", "a", "b", "b", "a", "b", "a", "b"))
  )
}

expect_prediction_matrix <- function(pred, n_rows) {
  expect_equal(nrow(pred), n_rows)
  expect_true(ncol(pred) >= 1L)
}

expect_skcla_base <- function(model, cls) {
  expect_s3_class(model, cls)
  expect_identical(model$attribute, "class")
  expect_identical(model$slevels, c("a", "b"))
}

test_that("sklearn classifier constructors expose expected state", {
  expect_skcla_base(skcla_knn("class", c("a", "b")), "skcla_knn")
  expect_identical(skcla_knn("class", c("a", "b"))$n_neighbors, 5L)

  expect_skcla_base(skcla_nb("class", c("a", "b")), "skcla_nb")
  expect_equal(skcla_nb("class", c("a", "b"))$var_smoothing, 1e-9)

  expect_skcla_base(skcla_rf("class", c("a", "b")), "skcla_rf")
  expect_identical(skcla_rf("class", c("a", "b"))$n_estimators, 100L)

  expect_skcla_base(skcla_gb("class", c("a", "b")), "skcla_gb")
  expect_identical(skcla_gb("class", c("a", "b"))$n_estimators, 100L)

  expect_skcla_base(skcla_svc("class", c("a", "b")), "skcla_svc")
  expect_identical(skcla_svc("class", c("a", "b"))$kernel, "rbf")

  expect_skcla_base(skcla_mlp("class", c("a", "b")), "skcla_mlp")
  expect_identical(skcla_mlp("class", c("a", "b"))$max_iter, 200L)
})

test_that("skcla_knn fits and predicts", {
  skip_if_no_sklearn()
  df <- make_skcla_data()
  model <- skcla_knn("class", c("a", "b"), n_neighbors = 3L)
  fitted <- fit.skcla_knn(model, df)
  pred <- predict.skcla_knn(fitted, df[, c("x1", "x2")])

  expect_prediction_matrix(pred, nrow(df))
  expect_true(!is.null(fitted$model))
})

test_that("skcla_nb fits and predicts", {
  skip_if_no_sklearn()
  df <- make_skcla_data()
  model <- skcla_nb("class", c("a", "b"))
  fitted <- fit.skcla_nb(model, df)
  pred <- predict.skcla_nb(fitted, df[, c("x1", "x2")])

  expect_prediction_matrix(pred, nrow(df))
  expect_true(!is.null(fitted$model))
})

test_that("skcla_rf fits and predicts", {
  skip_if_no_sklearn()
  df <- make_skcla_data()
  model <- skcla_rf("class", c("a", "b"), n_estimators = 10L)
  fitted <- fit.skcla_rf(model, df)
  pred <- predict.skcla_rf(fitted, df[, c("x1", "x2")])

  expect_prediction_matrix(pred, nrow(df))
  expect_true(!is.null(fitted$model))
})

test_that("skcla_gb fits and predicts", {
  skip_if_no_sklearn()
  df <- make_skcla_data()
  model <- skcla_gb("class", c("a", "b"), n_estimators = 10L)
  fitted <- fit.skcla_gb(model, df)
  pred <- predict.skcla_gb(fitted, df[, c("x1", "x2")])

  expect_prediction_matrix(pred, nrow(df))
  expect_true(!is.null(fitted$model))
})

test_that("skcla_svc fits and predicts", {
  skip_if_no_sklearn()
  df <- make_skcla_data()
  model <- skcla_svc("class", c("a", "b"), kernel = "linear")
  fitted <- fit.skcla_svc(model, df)
  pred <- predict.skcla_svc(fitted, df[, c("x1", "x2")])

  expect_prediction_matrix(pred, nrow(df))
  expect_true(!is.null(fitted$model))
})

test_that("skcla_mlp fits and predicts", {
  skip_if_no_sklearn()
  df <- make_skcla_data()
  model <- skcla_mlp("class", c("a", "b"), hidden_layer_sizes = c(4L), max_iter = 20L)
  fitted <- fit.skcla_mlp(model, df)
  pred <- predict.skcla_mlp(fitted, df[, c("x1", "x2")])

  expect_prediction_matrix(pred, nrow(df))
  expect_true(!is.null(fitted$model))
})
