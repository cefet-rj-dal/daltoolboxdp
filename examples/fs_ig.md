---
title: An R Markdown document converted from "Rmd/examples/fs_ig.ipynb"
output: html_document
---

## Feature Selection


```r
# DAL ToolBox
# version 1.0.727

source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox-examples/main/jupyter.R")
```

```
## Warning in file(filename, "r", encoding = encoding): cannot open URL
## 'https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox-examples/main/jupyter.R': HTTP status was '404 Not
## Found'
```

```
## Error in file(filename, "r", encoding = encoding): cannot open the connection to 'https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox-examples/main/jupyter.R'
```

```r
#loading DAL
load_library("daltoolbox") 
```

```
## Error in load_library("daltoolbox"): could not find function "load_library"
```

```r
load_github("cefet-rj-dal/tspredit")
```

```
## Error in load_github("cefet-rj-dal/tspredit"): could not find function "load_github"
```

### Example
General function for exploring feature selection methods


```r
iris <- datasets::iris
```

### IG


```r
myfeature <- fit(fs_ig("Species"), iris)
```

```
## Error in fit(fs_ig("Species"), iris): could not find function "fit"
```

```r
print(myfeature$features)
```

```
## Error in eval(expr, envir, enclos): object 'myfeature' not found
```

```r
data <- transform(myfeature, iris)
```

```
## Error in eval(expr, envir, enclos): object 'myfeature' not found
```

```r
print(head(data))
```

```
##                                                                             
## 1 function (..., list = character(), package = NULL, lib.loc = NULL,        
## 2     verbose = getOption("verbose"), envir = .GlobalEnv, overwrite = TRUE) 
## 3 {                                                                         
## 4     fileExt <- function(x) {                                              
## 5         db <- grepl("\\\\.[^.]+\\\\.(gz|bz2|xz)$", x)                     
## 6         ans <- sub(".*\\\\.", "", x)
```

