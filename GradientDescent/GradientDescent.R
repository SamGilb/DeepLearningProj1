source("GradientDescentHelper.R")

main <- function()
{
  #initialize
  numIterVector = 1:(maxIterations + 1)
  
  #############################################################################
  #############################################################################
  #                             First Data Set                                #
  #############################################################################
  #############################################################################
  #import data as a list
  import_data <- fread('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data')
  #convert list to matrix
  import_data <- matrix( unlist(import_data), nrow(import_data), ncol(import_data) )
  
  #randomized list of 0's 1's and 2's
  classificationList <- createClassificationList(import_data)
  
  #make X and Y vector/matrix
  y <- import_data[,58]
  X <- import_data[,1:57]
  X <- scale(X)
  
  #separate data using classificationList as needed
  X_train <- X[classificationList == 0,]
  X_validation <- X[classificationList == 1,]
  X_test <- X[classificationList == 2,]
  
  #separate y using 
  y_train <- y[classificationList == 0]
  y_validation <- y[classificationList == 1]
  y_test <- y[classificationList == 2]
  
  #print stuff for the report
  print("0     1")
  print(c("test",length(y_test[y_test==0]),length(y_test[y_test==1])))
  print(c("train",length(y_train[y_train==0]),length(y_train[y_train==1])))
  print(c("validation",length(y_validation[y_validation==0]),length(y_validation[y_validation==1])))
  
  #store weightMatrix
  weightMatrix <- GradientDescent( X_train, y_train, stepSize, maxIterations )
  
  #initialize vector variables
  trainErrorVector = NA
  validationErrorVector = NA
  testErrorVector = NA
  trainMeanLogLossVector = NA
  validationMeanLogLossVector = NA 
  #update vector variables
  for(i in 1:(maxIterations+1))
  {
    trainErrorVector[i] <- 1 - findAccuracy( X_train, y_train, weightMatrix[,i])
    validationErrorVector[i] <- 1 - findAccuracy( X_validation, y_validation, weightMatrix[,i] )
    testErrorVector[i] <- 1 - findAccuracy( X_test, y_test, weightMatrix[,i])
    trainMeanLogLossVector[i] <- calculateMeanLogLoss( X_train, y_train, weightMatrix[,i] )
    validationMeanLogLossVector[i] <- calculateMeanLogLoss( X_validation, y_validation, weightMatrix[,i] )
  }
  
  #print stuff for tables
  trainMinError = 1 - findAccuracy(X_train, y_train, weightMatrix[,which.min(trainErrorVector)])
  validationMinError = 1 - findAccuracy(X_validation, y_validation, weightMatrix[,which.min(validationErrorVector)])
  testMinError = 1 - findAccuracy(X_test, y_test, weightMatrix[,which.min(testErrorVector)])
  print("min log error for train, validation, test")
  print(trainMinError)
  print(validationMinError)
  print(testMinError)
  print("baseline for train, validation, test (subtract from 1 if needed)")
  print(length(y_train[y_train==1])/length(y_train))
  print(length(y_validation[y_validation==1])/length(y_validation))
  print(length(y_test[y_test==1])/length(y_test))
  
  #plot how %error and mean log loss for data set
  programPlot(numIterVector,trainErrorVector,validationErrorVector,trainMeanLogLossVector,validationMeanLogLossVector, weightMatrix)
  
  #############################################################################
  #############################################################################
  #                              Second Data Set                              #
  #############################################################################
  #############################################################################
  #import data as a list
  import_data <- fread('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data')
  import_data[,6][import_data[,6] == "Present"] = 1
  import_data[,6][import_data[,6] == "Absent"] = 0
  #convert list to matrix
  import_data <- matrix( as.numeric(unlist(import_data)), nrow(import_data), ncol(import_data) )
  
  #randomized list of 0's 1's and 2's
  classificationList <- createClassificationList(import_data)
  
  #make X and Y vector/matrix
  y <- import_data[,ncol(import_data)]
  X <- import_data[,2:10]
  X <- clean(X)
  X <- scale(X)
  
  #separate data using classificationList as needed
  X_train <- X[classificationList == 0,]
  X_validation <- X[classificationList == 1,]
  X_test <- X[classificationList == 2,]
  
  #separate y using 
  y_train <- y[classificationList == 0]
  y_validation <- y[classificationList == 1]
  y_test <- y[classificationList == 2]
  
  #print stuff for report
  print("0     1")
  print(c("test",length(y_test[y_test==0]),length(y_test[y_test==1])))
  print(c("train",length(y_train[y_train==0]),length(y_train[y_train==1])))
  print(c("validation",length(y_validation[y_validation==0]),length(y_validation[y_validation==1])))
  
  #store weightMatrix
  weightMatrix <- GradientDescent( X_train, y_train, stepSize, maxIterations )
  
  #initialize vector variables
  trainErrorVector = NA
  validationErrorVector = NA
  testErrorVector = NA
  trainMeanLogLossVector = NA
  validationMeanLogLossVector = NA
  #update vector variables
  for(i in 1:(maxIterations+1))
  {
    trainErrorVector[i] <- 1 - findAccuracy( X_train, y_train, weightMatrix[,i])
    validationErrorVector[i] <- 1 - findAccuracy( X_validation, y_validation, weightMatrix[,i] )
    testErrorVector[i] <- 1 - findAccuracy( X_test, y_test, weightMatrix[,i])
    trainMeanLogLossVector[i] <- calculateMeanLogLoss( X_train, y_train, weightMatrix[,i] )
    validationMeanLogLossVector[i] <- calculateMeanLogLoss( X_validation, y_validation, weightMatrix[,i] )
  }
  
  #print stuff for tables
  trainMinError = 1 - findAccuracy(X_train, y_train, weightMatrix[,which.min(trainErrorVector)])
  validationMinError = 1 - findAccuracy(X_validation, y_validation, weightMatrix[,which.min(validationErrorVector)])
  testMinError = 1 - findAccuracy(X_test, y_test, weightMatrix[,which.min(testErrorVector)])
  print("min log error for train, validation, test")
  print(trainMinError)
  print(validationMinError)
  print(testMinError)
  print("baseline for train, validation, test (subtract from 1 if needed)")
  print(length(y_train[y_train==1])/length(y_train))
  print(length(y_validation[y_validation==1])/length(y_validation))
  print(length(y_test[y_test==1])/length(y_test))
  
  #plot how %error and mean log loss for data set
  programPlot(numIterVector,trainErrorVector,validationErrorVector,trainMeanLogLossVector,validationMeanLogLossVector, weightMatrix)
  
  #############################################################################
  #############################################################################
  #                             Third Data Set                                #
  #############################################################################
  #############################################################################
  #import data as a list
  import_data <- fread('zip.train')
  import_data <- matrix(as.numeric(unlist(import_data)), nrow(import_data), ncol(import_data))
  import_data <- rbind(import_data[import_data[,1] == 0,], import_data[import_data[,1] == 1,])
  
  #randomized list of 0's 1's and 2's
  classificationList <- createClassificationList(import_data)
  
  #make X and Y vector/matrix
  y <- import_data[,1]
  X <- import_data[,2:257]
  X <- clean(X)
  X <- scale(X)
  
  #separate data using classificationList as needed
  X_train <- X[classificationList == 0,]
  X_validation <- X[classificationList == 1,]
  X_test <- X[classificationList == 2,]
  
  #separate y using 
  y_train <- y[classificationList == 0]
  y_validation <- y[classificationList == 1]
  y_test <- y[classificationList == 2]
  
  #print stuff for report
  print("0     1")
  print(c("test",length(y_test[y_test==0]),length(y_test[y_test==1])))
  print(c("train",length(y_train[y_train==0]),length(y_train[y_train==1])))
  print(c("validation",length(y_validation[y_validation==0]),length(y_validation[y_validation==1])))
  
  #store weightMatrix
  weightMatrix <- GradientDescent( X_train, y_train, stepSize, maxIterations )
  
  #initialize vector variables
  trainErrorVector = NA
  validationErrorVector = NA
  testErrorVector = NA
  trainMeanLogLossVector = NA
  validationMeanLogLossVector = NA 
  #update vector variables
  for(i in 1:(maxIterations+1))
  {
    trainErrorVector[i] <- 1 - findAccuracy( X_train, y_train, weightMatrix[,i])
    validationErrorVector[i] <- 1 - findAccuracy( X_validation, y_validation, weightMatrix[,i] )
    testErrorVector[i] <- 1 - findAccuracy( X_test, y_test, weightMatrix[,i])
    trainMeanLogLossVector[i] <- calculateMeanLogLoss( X_train, y_train, weightMatrix[,i] )
    validationMeanLogLossVector[i] <- calculateMeanLogLoss( X_validation, y_validation, weightMatrix[,i] )
  }
  
  #print stuff for pdf
  trainMinError = 1 - findAccuracy(X_train, y_train, weightMatrix[,which.min(trainErrorVector)])
  validationMinError = 1 - findAccuracy(X_validation, y_validation, weightMatrix[,which.min(validationErrorVector)])
  testMinError = 1 - findAccuracy(X_test, y_test, weightMatrix[,which.min(testErrorVector)])
  print("min log error for train, validation, test")
  print(trainMinError)
  print(validationMinError)
  print(testMinError)
  print("baseline for train, validation, test (subtract from 1 if needed)")
  print(length(y_train[y_train==1])/length(y_train))
  print(length(y_validation[y_validation==1])/length(y_validation))
  print(length(y_test[y_test==1])/length(y_test))
  
  #plot how %error and mean log loss for data set
  programPlot(numIterVector,trainErrorVector,validationErrorVector,trainMeanLogLossVector,validationMeanLogLossVector, weightMatrix)
  
  #Return a weightMatrix
  weightMatrix
}

#run this big boy
weightMatrix <- main()

