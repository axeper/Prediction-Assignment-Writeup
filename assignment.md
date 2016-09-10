# Prediction Assignment Writeup
Axel Perruchoud  
8 septembre 2016  



## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement â a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify __how much__ of a particular activity they do, but they rarely quantify __how well__ they do it. 

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)



## Goal of the Assignment

The goal of the project is to predict the manner in which they did the exercise. This is the `classe` variable in the training set. You may use any of the other variables to predict with. You should create a report describing:

* How the model was built
* How cross validation was used
* What you think the expected out of sample error is
* Why you made the choices you did

The prediction model will be used to predict 20 different test cases.



## Exploratory part

After loading the data, we can display some information to learn more about it.


```r
    data <- read.csv("pml-training.csv")
    dim(data)
```

```
## [1] 19622   160
```

```r
    summary(data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

As we can see, we have a lot of variables and a lot of observations. As our goal is to predict the classe (i.e. `data$classe`), we need first to do some cleanup.


## Preprocessing

Let's split the data into a train and a test set.


```r
    library(caret)

    set.seed(1234)
	inTrain <- createDataPartition(y = data$classe, p = 0.7, list = FALSE)
	training <- data[inTrain,]
	testing <- data[-inTrain,]
	
	str(training, list.len = 20)
```

```
## 'data.frame':	13737 obs. of  160 variables:
##  $ X                       : int  2 3 4 5 6 7 8 9 10 11 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  808298 820366 120339 196328 304277 368296 440390 484323 484434 500302 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 8.18 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##   [list output truncated]
```

When looking at the above results, there are multiple issues:

1. Some columns are not useful (`X`, `user_name`, etc) and can be removed.
2. Some columns are factor instead of numeric.
3. Some columns are full of NAs. 

Let's remediate this with a custom preprocess function.


```r
    preprocess <- function(df) {
        
        # Remove the useless first columns
        df1 <- df[,-c(1:7)]
        
        # Convert factors to numeric (but not the last column)
        indx <- sapply(df1[ , -length(df1)], is.factor)
        df1[indx] <- lapply(df1[indx], function(x) as.numeric(as.character(x)))
        
        # Remove columns which are 90% NAs or more
        sumNAs <- apply(df1, 2, function(x) sum(is.na(x)))  # Can use mean(is.na())
        tooManyNA <- (sumNAs / dim(df1)[1]) > .9
        df2 <- df1[, !tooManyNA]
        df2
    } 

    # Preprocess the data
    trainingClean <- preprocess(training)
    testingClean <- preprocess(testing)
    dim(trainingClean)
```

```
## [1] 13737    53
```

We reduced the data from 160 to 53 columns.


## Model selection

### Simple Linear Model
We first want to try a simple linear model. Let's look at the correlations with `classe`.


```r
    corr <- cor(x = trainingClean[ , -length(trainingClean)], y = as.numeric(trainingClean$classe))
    head(sort(abs(corr), decreasing = TRUE))
```

```
## [1] 0.3463655 0.2934925 0.2872298 0.2467784 0.2368272 0.1854378
```

Looking at the results above, the highest correlation is about $0.34$. This is not high enough to hope good results, but let's try nevertheless.


```r
    ggplot(data = trainingClean, aes(classe, pitch_forearm)) + geom_boxplot(aes(fill = classe))
```

![](assignment_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

The boxplot above confirms that there is no easy linear separation between `classe` and its highest correlated variable, `pitch_forearm`.

### rpart model

We want to evaluate a rpart model with cross-validation.


```r
    library(rpart)
    train_control<- trainControl(method = "cv", number = 10, savePredictions = TRUE)
    rpartFit <- train(classe ~ ., data = trainingClean, method = 'rpart', trControl = train_control)
    confusionMatrix(testingClean$classe, predict(rpartFit, testingClean))$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1530   35  105    0    4
##          B  486  379  274    0    0
##          C  493   31  502    0    0
##          D  452  164  348    0    0
##          E  168  145  302    0  467
```

This model show a disappointing performance: the prediction table is all over the place. Next method. 


### Random Forest Model

Finally, let's try random forest with `ntree` set to 100 and with different preprocessing.


```r
    Y = trainingClean$classe
    X = trainingClean[ , -length(trainingClean)]
    rfFit1 <- train(y = Y, x = X, method = 'rf', ntree = 100)
    rfFit2 <- train(y = Y, x = X, method = 'rf', ntree = 100, trControl = train_control)
    rfFit3 <- train(y = Y, x = X, method = 'rf', ntree = 100, preProcess = "pca")
    rfFit4 <- train(y = Y, x = X, method = 'rf', ntree = 100, trControl = train_control, preProcess = "pca")
```

Our four models are: 

* `rfFit1`, no train preprocessing
* `rfFit2`, with cross-validation
* `rfFit3`, with PCA
* `rfFit4`, with crossvalidation and PCA


### Model Evaluation

Let's compare the accuracy of each model.


```r
    c(rfFit1.accuracy = max(rfFit1$results$Accuracy),
      rfFit2.accuracy = max(rfFit2$results$Accuracy),
      rfFit3.accuracy = max(rfFit3$results$Accuracy),
      rfFit4.accuracy = max(rfFit4$results$Accuracy))
```

```
## rfFit1.accuracy rfFit2.accuracy rfFit3.accuracy rfFit4.accuracy 
##       0.9880771       0.9925744       0.9577999       0.9697178
```
Quite amazingly, we have a 99\% accuracy for `rfFit2` as our best model.

Also, we can notice that the PCA did not perform better than the non-PCA model, which was to be expected as PCA is a form of data compression. Finally, the cross-validation models (`rfFit2` and `rfFit4`) performed better than the other models.


### Conclusion

Our final model then is `rfFit2` with `mtry` set to $27$. The model was simply built using random forest with $100$ trees and cross-validation.


```r
    rfFit2
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12363, 12362, 12363, 12363, 12365, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9918475  0.9896860
##   27    0.9925744  0.9906059
##   52    0.9894436  0.9866450
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
    confusionMatrix(testingClean$classe, predict(rfFit2, testingClean))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   13 1125    1    0    0
##          C    0    3 1019    4    0
##          D    0    0    7  956    1
##          E    0    0    2    4 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9917, 0.9959)
##     No Information Rate : 0.2867          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9925          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9923   0.9973   0.9903   0.9917   0.9991
## Specificity            1.0000   0.9971   0.9986   0.9984   0.9988
## Pos Pred Value         1.0000   0.9877   0.9932   0.9917   0.9945
## Neg Pred Value         0.9969   0.9994   0.9979   0.9984   0.9998
## Prevalence             0.2867   0.1917   0.1749   0.1638   0.1830
## Detection Rate         0.2845   0.1912   0.1732   0.1624   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9961   0.9972   0.9944   0.9950   0.9989
```

Given the very high accuracy, the model should have an error rate lower than $5\%$, we can expect our $20$ predictions to be accurate.


### Test Results

For the test results, we can use the `predict()` function with `rfFit2`. We can also display the results predicted by the other models for curiosity.


```r
    testingFinal <- preprocess(read.csv("pml-testing.csv"))
    t(cbind(
        rfFit2 = as.data.frame(predict(rfFit2, newdata = testingFinal[, -length(testingFinal)])),
        rfFit1 = as.data.frame(predict(rfFit1, newdata = testingFinal[, -length(testingFinal)])),
        rfFit3 = as.data.frame(predict(rfFit3, newdata = testingFinal[, -length(testingFinal)])),
        rfFit4 = as.data.frame(predict(rfFit4, newdata = testingFinal[, -length(testingFinal)]))))
```

Note: the results are not displayed voluntarily.
