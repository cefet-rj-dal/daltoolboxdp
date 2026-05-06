library(testthat)
library(daltoolboxdp)

run_integration_tests <- tolower(Sys.getenv("DALTOOLBOXDP_RUN_INTEGRATION_TESTS", "false")) %in% c("true", "1", "yes")

if (run_integration_tests) {
  if (dir.exists("tests/testthat")) {
    test_dir("tests/testthat")
  } else if (dir.exists("testthat")) {
    test_dir("testthat")
  } else {
    test_check("daltoolboxdp")
  }
} else {
  cat("Skipping daltoolboxdp integration tests. Set DALTOOLBOXDP_RUN_INTEGRATION_TESTS=true to run them.\n")
}
