# Feature Selection

# Installation (if needed)
#install.packages("daltoolboxdp")

# Loading packages
library(daltoolbox)
library(daltoolboxdp)

# Example data
iris <- datasets::iris

# FSS - step by step

# 1) Fit the selector with target "Species"
myfeature <- fit(fs_fss("Species"), iris)

# 2) View selected features
print(myfeature$features)

# 3) Transform data to keep selected features + target
data <- transform(myfeature, iris)
print(head(data))
