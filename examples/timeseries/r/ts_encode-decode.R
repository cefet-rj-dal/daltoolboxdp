# Loading required packages
library(daltoolbox)

data(tsd)
tsd$y[39] <- tsd$y[39] * 6   # inject a synthetic outlier for illustration

sw_size <- 5                         # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)        # series -> windows with p columns
ts_head(ts, 3)

library(ggplot2)
plot_ts(x = tsd$x, y = tsd$y) +
  theme(text = element_text(size = 16))

samp <- ts_sample(ts, test_size = 5)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)

auto <- autoenc_ed(5, 3)             # 5 -> 3 -> 5 dimensions
auto <- fit(auto, train)

print(head(train))                    # original windows (p columns)
result <- transform(auto, train)      # reconstructed windows (p columns)
print(head(result))

print(head(test))
result <- transform(auto, test)
print(head(result))
