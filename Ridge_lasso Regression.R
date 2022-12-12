library(plyr)
library(readr)
library(dplyr)
library(caret)
library(glmnet)
#importing data
study_data<-read.csv("RealEstate.csv", sep=",", header=T, row.names = NULL)
study_data<-as.data.frame(study_data)
n=781#no. of variables
#one-hot encoding status column
status_col<-as.data.frame(study_data[,8], drop=FALSE)#extracting status column
dmy<-dummyVars("~.", data=status_col, fullRank=F)#creating dummies
status_col.transformed<-data.frame(predict(dmy,newdata=status_col))#one-hot encoding dummies
glimpse(status_col.transformed)
#one-hot encoding location column
location_col<-as.data.frame(study_data[,2], drop=FALSE)#extracting status column
dmy<-dummyVars("~.", data=location_col, fullRank=F)#creating dummies
location_col.transformed<-data.frame(predict(dmy,newdata=location_col))#one-hot encoding dummies
glimpse(location_col.transformed)
location_col.transformed<-as.data.frame(location_col.transformed)
#data transformation
study_data1<-as.data.frame(study_data[,1], drop=FALSE)#extracting column 1
study_data2<-as.data.frame(study_data[,3:7], drop=FALSE)#extracting columns 3-7
study_data.main<-c(study_data1,location_col.transformed, study_data2, status_col.transformed)#creating new data frame with one-hot encoded status column
study_data.main<-as.data.frame(study_data.main)
glimpse(study_data.main)
#splitting into x and y
y<-as.data.frame(study_data.main[,56], drop=FALSE)#extracting price as vector (price column no. is 56)
x<-as.data.frame(c(study_data.main[,1:55],study_data.main[,57:63]), drop=FALSE)#extracting predictor variables (rest of the columns apart from 56)
glimpse(x)                 
glimpse(y)
#dividing into train and testing sets
train_rows<-sample(1:n, 0.75*n)
x.train<-x[train_rows, ]
x.test<-x[-train_rows, ]
y.train<-y[train_rows, ]
y.test<-y[-train_rows, ]
#RIDGE REGRESSION (alpha=0)
x.test<-as.matrix(x.test)
x.train<-as.matrix(x.train)
alpha0.fit<-cv.glmnet(x.train, y.train, type.measure="mse", alpha=0, lambda.min=0, lambda.max=1000000, nfolds=5, family="gaussian")#Ridge regression model using 5-fold cross validation
alpha0.predicted<-predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)
mse_1<-mean((y.test-alpha0.predicted)^2)#CHECK
rss_1<-mse_1*n
alpha0.fit$lambda.min
plot(alpha0.fit$lambda)
plot(alpha0.fit$cvm)
#locator()#for testing
summary(alpha0.fit)
#extra plots
plot(alpha0.fit$glmnet.fit)
plot(y.test, alpha0.predicted, col=c("black","red"))
mean((y.test-alpha0.predicted)^2)
#LASSO REGRESSION (alpha=1)
alpha1.fit<-cv.glmnet(x.train, y.train, type.measure = "mse", alpha=1, nfolds=5, family="gaussian")
alpha1.predicted<-predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)
mse_2<-mean((y.test-alpha1.predicted)^2)
rss_2<-mse_2*n
plot(alpha1.fit$lambda, alpha1.fit$cvlo)
plot(alpha0.fit$lambda, alpha0.fit$cvlo)
plot(alpha1.fit$cvm)
plot(y.test, alpha1.predicted, col=c("black","red"))
summary(alpha0.fit)
