---
title: An R Markdown document converted from "Rmd/examples/bal_subsampling.ipynb"
output: html_document
---

## Feature Selection


```r
# TSPredIT
# version 1.0.727

source("https://raw.githubusercontent.com/cefet-rj-dal/tspredit-examples/main/jupyter.R")
```

```
## Warning in file(filename, "r", encoding = encoding): cannot open URL
## 'https://raw.githubusercontent.com/cefet-rj-dal/tspredit-examples/main/jupyter.R': HTTP status was '404 Not Found'
```

```
## Error in file(filename, "r", encoding = encoding): cannot open the connection to 'https://raw.githubusercontent.com/cefet-rj-dal/tspredit-examples/main/jupyter.R'
```

```r
#loading DAL
load_library("daltoolbox") 
```

```
## Error in load_library("daltoolbox"): could not find function "load_library"
```

```r
load_library("tspredit")
```

```
## Error in load_library("tspredit"): could not find function "load_library"
```

### Example
General function for exploring feature selection methods


```r
iris <- datasets::iris
data(iris)
mod_iris <- iris[c(1:50,51:71,101:111),]
table(mod_iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```

### subsampling


```r
bal <- bal_subsampling('Species')
```

```
## Error in bal_subsampling("Species"): could not find function "bal_subsampling"
```

```r
bal <- daltoolbox::fit(bal, mod_iris)
```

```
## Error in eval(expr, envir, enclos): object 'bal' not found
```

```r
adjust_iris <- daltoolbox::transform(bal, mod_iris)
```

```
## Error in eval(expr, envir, enclos): object 'bal' not found
```

```r
table(adjust_iris$Species)
```

```
## Error in eval(expr, envir, enclos): object 'adjust_iris' not found
```

