# Loading required packages
library(daltoolbox)

data(tsd)
tsd$y[39] <- tsd$y[39] * 6   # inject a synthetic outlier for illustration in the plot

sw_size <- 5                         # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)        # convert the series into windows with p columns
ts_head(ts, 3)                       # view the first 3 windows

library(ggplot2)
plot_ts(x = tsd$x, y = tsd$y) +      # series plot with the outlier peak
  theme(text = element_text(size = 16))

samp <- ts_sample(ts, test_size = 5) # hold out the last 5 windows for test
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)

auto <- autoenc_e(5, 3)              # reduce from 5 -> 3 dimensions (p -> k)
auto <- fit(auto, train)

print(head(train))                    # original windows (p columns)
result <- transform(auto, train)      # encodings (k columns)
print(head(result))

print(head(test))
result <- transform(auto, test)
print(head(result))
