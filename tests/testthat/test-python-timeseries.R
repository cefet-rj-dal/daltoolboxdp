make_ts_data <- function(n = 16L) {
  x <- data.frame(
    x1 = seq_len(n),
    x2 = seq_len(n) + 1,
    x3 = seq_len(n) + 2
  )
  y <- seq_len(n) * 0.5
  list(x = x, y = y)
}

test_that("ts_lstm constructor exposes unified defaults", {
  model <- ts_lstm(input_size = 3L)

  expect_s3_class(model, "ts_lstm")
  expect_identical(model$epochs, 100L)
  expect_identical(model$validation_strategy, "static")
  expect_identical(model$stopping_rule, "none")
})

test_that("ts_lstm fits and predicts through the R wrapper", {
  skip_if_no_pytorch()

  data <- make_ts_data()
  model <- ts_lstm(input_size = 3L, epochs = 2L)
  fitted <- do_fit.ts_lstm(model, data$x, data$y)
  pred <- do_predict.ts_lstm(fitted, data$x)

  expect_length(pred, nrow(data$x))
  expect_true(length(fitted$train_loss_hist) >= 1)
  expect_true(!is.null(fitted$model))
})

test_that("ts_conv1d constructor exposes unified defaults", {
  model <- ts_conv1d(input_size = 3L)

  expect_s3_class(model, "ts_conv1d")
  expect_identical(model$epochs, 100L)
  expect_identical(model$validation_strategy, "static")
  expect_identical(model$stopping_rule, "none")
})

test_that("ts_conv1d fits and predicts through the R wrapper", {
  skip_if_no_pytorch()

  data <- make_ts_data()
  model <- ts_conv1d(
    input_size = 3L,
    epochs = 2L,
    validation_strategy = "dynamic",
    stopping_rule = "ema"
  )
  fitted <- do_fit.ts_conv1d(model, data$x, data$y)
  pred <- do_predict.ts_conv1d(fitted, data$x)

  expect_length(pred, nrow(data$x))
  expect_true(length(fitted$train_loss_hist) >= 1)
  expect_true(!is.null(fitted$model))
})
