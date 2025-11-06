# Installation (if needed)
#install.packages("daltoolboxdp")

# Loading packages
library(daltoolbox)
library(daltoolboxdp)

# Example data
iris <- datasets::iris

# Information Gain (IG) - step by step

# 1) Fit the feature selector (target: Species)
myfeature <- fit(fs_ig("Species"), iris)

# 2) View selected features
print(myfeature$features)

# 3) Apply transformation to keep only selected features + target
data <- transform(myfeature, iris)
print(head(data))
