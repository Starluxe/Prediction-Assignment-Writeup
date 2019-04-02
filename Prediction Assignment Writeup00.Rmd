---
title: "Prediction Assignment Writeup"
author: "Luciano Guerra"
date: "April 1, 2019"
output: html_document
---

# Prediction Assignment Writeup - Project Assignment

## Background  

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data  

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Peer Review Portion  
Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on gradees.

### Course Project Prediction Quiz Portion  
Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading.

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Data Processing

### Libraries needed and download data. Clean the data as well.

```{r echo=TRUE, message=FALSE}
# library
library(caret)
library(ggplot2)
library(rattle)

library(rpart.plot)
library(corrplot)
library(randomForest)
library(RColorBrewer)
setwd("E:/Luciano/R_WorkingDir/Project01")
rm(list = ls())
```

```{r echo=TRUE}
# donwloada data
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
csvTrain <- "pml-training.csv"
csvTest <-  "pml-testing.csv"

if (file.exists(csvTrain)) {
        training <- read.csv(csvTrain, na.strings=c("NA","#DIV/0!",""))
} else { 
        download.file(urlTrain,csvTrain)
        training <- read.csv(csvTrain, na.strings=c("NA","#DIV/0!",""))
}                           


if (file.exists(csvTest)) {
        testing <- read.csv(csvTest, na.strings=c("NA","#DIV/0!",""))
} else { 
        download.file(urlTest,csvTest)
        testing <- read.csv(csvTest, na.strings=c("NA","#DIV/0!",""))
}  
dim(training)
dim(testing)
sum(is.na(training))
rm(urlTrain)
rm(urlTest)

```
Data sets are as follow:
* Training data
    + Obesrvations: `r dim(training)[1]` 
    + Variables:    `r dim(training)[2]` 
* Testing data
    + Obesrvations: `r dim(testing)[1]`
    + Variables:    `r dim(testing)[2]` 

### Cleaning data  
#### Remove **Near to Zero varianze (NZV)** variables 
```{r echo=TRUE}
NZV <- nearZeroVar(training, saveMetrics = TRUE)
head(NZV, 20)
trainingNZV <- training[, !NZV$nzv]
testingNZV <- testing[, !NZV$nzv]
dim(trainingNZV)
dim(testingNZV)
rm(training)
rm(testing)
rm(NZV)
```
#### Remove some other columns with no information
```{r echo=TRUE}
noInfoCol <- grepl("^X|timestamp|user_name", names(trainingNZV))
trainingNZVCol <- trainingNZV[, !noInfoCol]
testingNZVCol <- testingNZV[, !noInfoCol]
rm(noInfoCol)
rm(trainingNZV)
rm(testingNZV)
dim(trainingNZVCol)
dim(testingNZVCol)
```  

#### Remove columns with `NA's`.
```{r echo=TRUE}
NACols <- (colSums(is.na(trainingNZVCol)) == 0)
trainingNew <- trainingNZVCol[, NACols]
testingNew <- testingNZVCol[, NACols]
rm(trainingNZVCol)
rm(testingNZVCol)
rm(NACols)
``` 
Data sets updated are as follow:
* Training data
    + Obesrvations: `r dim(trainingNew)[1]` 
    + Variables:    `r dim(trainingNew)[2]` 
* Testing data
    + Obesrvations: `r dim(testingNew)[1]`
    + Variables:    `r dim(testingNew)[2]`

### Partitioning the training set into two

Partioning Training data set into two data sets, 60% for newTraining, 40% for newTesting
```{r echo=FALSE}
inTrain <- createDataPartition(trainingNew$classe, p=0.6, list = FALSE)
newTraining <- trainingNew[inTrain,]
newTesting <- trainingNew[-inTrain,]
dim(newTraining)
dim(newTesting)
rm(inTrain)
rm(trainingNew)
```
The Dataset now consists of `r dim(newTraining)[2]` variables with the observations divided as following:  
1. Training Data: `r dim(newTraining)[1]` observations.  
2. Validation Data: `r dim(newTesting)[1]` observations.  
3. Testing Data: `r dim(testingNew)[1]` observations.  

## Data Modeling - ML algorithms for prediction 
### Decision Tree
```{r echo=TRUE}
modFitDT <- rpart(classe ~., data = newTraining, method = "class")
fancyRpartPlot(modFitDT)
```
#### Predicting
```{r echo=FALSE}
predModFitDT <- predict(modFitDT, newTesting, type = "class")
confusionMatrix(newTesting$classe, predModFitDT)
```

### Random Forest  
We fit a predictive model for activity recognition using <b>Random Forest</b> algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general.  
We will use <b>5-fold cross validation</b> when applying the algorithm.  
```{r echo=TRUE, cache=TRUE}
modFitRF <- train(classe ~ ., data = newTraining, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modFitRF
```
#### Predicting  
```{r echo=TRUE}
predictRF <- predict(modFitRF, newTesting)
confusionMatrix(newTesting$classe, predictRF)
accuracy <- postResample(predictRF, newTesting$classe)
ose <- 1 - as.numeric(confusionMatrix(newTesting$classe, predictRF)$overall[1])
rm(predictRF)
```


The Estimated Accuracy of the Random Forest Model is `r accuracy[1]*100`% and the Estimated Out-of-Sample Error is `r ose*100`%.  
Random Forests yielded better Results, as expected!  

## Predicting The Manner of Exercise for Test Data Set  
Now, we apply the <b>Random Forest</b> model to the original testing data set downloaded from the data source. We remove the problem_id column first.  
```{r echo=TRUE}
rm(accuracy)
rm(ose)
predict(modFitRF, testingNew[, -length(names(testingNew))])
``` 

##  Files  for the Assignment -  Submission to Coursera

```{r warning=FALSE, error=FALSE}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./AssignementFiles/problem_id_",i,".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}

pml_write_files(predict(modFitRF, testingNew[, -length(names(testingNew))]))
rm(modFitRF)
rm(trainingNew)
rm(testingNew)
rm(newTesting)
rm(pml_write_files)
```

