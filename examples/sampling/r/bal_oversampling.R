# Installation (if needed)
#install.packages("daltoolboxdp")

# Loading packages
library(daltoolbox)
library(daltoolboxdp)

# Example data and creation of artificial imbalance
iris <- datasets::iris
data(iris)
mod_iris <- iris[c(1:50,51:71,101:111),]   # subset with imbalanced classes
table(mod_iris$Species)                     # original distribution

# Oversampling - increase the minority class to balance
bal <- bal_oversampling('Species')
bal <- daltoolbox::fit(bal, mod_iris)
adjust_iris <- daltoolbox::transform(bal, mod_iris)
table(adjust_iris$Species)                   # distribution after oversampling
