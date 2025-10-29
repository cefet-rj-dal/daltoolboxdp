# Installation (if needed)
#install.packages("daltoolboxdp")

# Loading packages
library(daltoolbox)
library(daltoolboxdp)

# Example data and creation of artificial imbalance
iris <- datasets::iris
data(iris)
mod_iris <- iris[c(1:50,51:71,101:111),]
table(mod_iris$Species)                      # original distribution

# Subsampling - reduce the majority class to balance
bal <- bal_subsampling('Species')
bal <- daltoolbox::fit(bal, mod_iris)
adjust_iris <- daltoolbox::transform(bal, mod_iris)
table(adjust_iris$Species)                    # distribution after subsampling
