
###############################################################################
###############################################################################
#                       Global Constants and Libraries                        #
###############################################################################
###############################################################################
library("sigmoid")
library("ggplot2")
library("data.table")
library("WeightedROC")

maxIterations = 100
stepSize = 0.5

###############################################################################
###############################################################################
#                           Helper Functions                                  #
###############################################################################
###############################################################################

#Function Name: GradientDescent
#Algorithm: Use the gradient descent method to optimize weights for a set of
#           training data
#Input(s)
#X: data matrix, each row represents one obersvation, each column represents
#   a feature
#Y: data vector, each entry represents a 0 or 1 for binary classification
#stepSize: a positive real number that controls how far to step
#          in the negative gradient direction
#maxIterations: positive integer that controls how many steps to take
#Output(s): 
#weightMatrix: returns a matrix called weightMatrix of real numbers where
#        the number of rows is the number of input features
#        the number of columns is maxIterations
GradientDescent <- function( X, y, stepSize, maxIterations )
{
  #initialization
  weightMatrix <- matrix(data = 0, nrow = ncol(X), ncol = 1 )
  
  #descend maxIteration times
  for( currIter in 1:maxIterations )
  {
    #find and store current weight
    weightVector <- weightMatrix[,currIter]
    
    #given the curren weight, and the data, find the new weight after
    #one iteration of gradient descent
    newWeight <- findDescent( X, y, stepSize, weightVector )
    
    #bind the new weight vector to weightMatrix
    weightMatrix <- cbind( weightMatrix, cbind(matrix(newWeight,ncol(X),1)) )
  }
  
  #return weightMatrix
  weightMatrix
}

#Function Name; findDescent
#Algorithm: This will simulate and return a "single descent" in a given 
#           GradientDescent based algorithm
#Input(S):
#X: data matrix, each row represents one obersvation, each column represents
#   a feature
#Y: data vector, each entry represents a 0 or 1 for binary classification
#stepSize: a positive real number that controls how far to step
#          in the negative gradient direction
#currWeight: most updated vector of weights
#Output(s):
#newWeight: current weight vector plus a step in the direction of the
#           negative gradient
findDescent <- function( X, y, stepSize, currWeight )
{
  
  #calculate gradient
  gradient <- calculateGradient( X, y, currWeight )
  
  #find newWeight or the "descent"
  newWeight <- (-1 * gradient * stepSize) + currWeight
  
  #return newWeight
  cbind(newWeight)
}

#FunctionName: findAccuracy
#Algorithm: Given test data that the machine has not seen before, and 
#           predicted weights, find the proportion of times the model
#           correctly predicts unseen output with the total number of
#           predictions
#Inputs:
#X: unseen data, each row represents an observation and each column
#   and each column represents a feature
#predWeight: predicted weight vector
#Y: Actual outputs for the unseen data, is compared to the product of
#   X and predicted weights
#Outputs: A number between 0 and 1 indicated the proportion of success
#         that our model has
#Note: This function only works for logistical regression because and
#      probably wont do much outside of the scope of this project
findAccuracy <-function( X, y, predWeight)
{
  #predict y values
  predicted_y = round(sigmoid(X %*% predWeight))
  
  #conditionally select where predicted_y == y, then see what the length
  #is of what is selected
  num_correct = length(y[predicted_y == y])
  num_total = length(y)
  
  #calculate accuracy
  accuracy <- num_correct / num_total
  
  #return
  accuracy
}

#Function Name: makeClassificationList
#Algorithm: This function will create an unordered list of 0's, 1's, and 2's.
#           The purpose of this is to later use this list for conditional
#           selection to separate data into 3 categories: train(0)
#           validation(1) and test(2). About 60% of it will be 0's, about 
#           20% will be 1's, and about 20% will be 2's.
#Inputs:
#X: data matrix, rows = observations, columns = features
createClassificationList <- function( X )
{
  #number of observations
  num_obs = nrow(X)
  
  #integer division plus remainder
  num_train = ((num_obs %/% 10) * 6) + num_obs %% 10
  
  #integer division
  num_validation = ((num_obs %/% 10) * 2)
  
  #integer division
  num_test = ((num_obs %/% 10) * 2)
  
  #create list of 0s and 1s
  classificationList <- c(replicate(num_train,0),replicate(num_validation,1),replicate(num_test,2))
  
  #randomize list
  classificationList <- sample(classificationList)
  
  #return
  classificationList
}

#Function Name: calculateGradient
#Algorithm: Calculates gradient for mean logistic loss given some data matrix, 
#           output vector, and predicted weights
#Input:
#X: data matrix, rows = observations features = columns
#y: output vector,
#predWeight: predicted weights so far
#Output
#gradient vector used for gradient descent algorithm
calculateGradient <- function( X, y, predWeight )
{
  #m = num of observations
  numObs = nrow(X)
  
  #find sum of gradients for individual trials
  sum = 0
  for( i in 1:numObs )
  {
    #initialization for readability
    theta <- cbind( predWeight )
    yTilda_i <- getYTilda(y)[i]
    x_i <- X[i,]
    
    
    #find gradient in 3 parts
    
    #part 1
    c1 <- 1 / (1 + exp(-1 * yTilda_i * t(theta) %*% x_i))
    #part 2
    c2 <- exp(-1 * yTilda_i * t(theta) %*% x_i)
    #part 3
    vector <- -1 %*% yTilda_i %*% x_i
    
    #calculate gradient for one log loss
    gradient <- c1 %*% c2 %*% vector
    
    sum = sum + gradient
  }
  
  #calculate mean
  mean = sum / numObs
  
  #return mean
  mean
}

#Function Name: calculateMeanLogLoss
#Algorithm: Calculates mean log loss for some given observation and current weight
#Input:
#x: data vector for a given observation
#y: scalar for a given output
#predWeight: predicted weights so far
calculateMeanLogLoss <- function( X, y, predWeight )
{
  #calculate number of observations
  numObs = nrow(X)
  
  #initialize sum so you can take average later
  sum = 0
  for(i in 1:numObs)
  {
    #add up all the log losses
    sum = sum + log(1 + exp(-1 * getYTilda(y)[i] * rbind(predWeight) %*% cbind(X[i,])))

  }
  
  #calculate mean
  mean = sum / numObs
  
  #return
  mean
}

#Function Name: getYTilda
#Algorithm: gets y~ as descriped in class, where 0 is mapped to -1
#           and 1 is mapped to 1
#Input
#y: output vector, indicating binary results
#Output
#returns y~
getYTilda <- function( y )
{
  #conditional selection and assignment
  y[y==0] = -1
  yTilda <- y
  
  #set y back to the original y, just for safety
  y[y==-1] = 0
  
  #return yTilda
  yTilda
}

#Function Name: programPlot
#Algorithm: simplifies the program plotting process for the training vs
#           validation % error and training vs validation mean log loss
#Inputs:
#NumIterVector: a list of 1:(maxIterations + 1), used for x-axis with plotting
#trainErrorVector: vector of training error where trainErrorVector[i] is
#                  the training error for the i'th observation
#validatoinErrorVector: vector of validation error where validatoinErrorVector[i]
#                       is the training error for the i'th observation
#trainMeanLogLossVector: vector of training mean log loss where 
#                        trainMeanLogLossVector[i] is the training error for
#                        the i'th observation
#validationMeanLogLossVector:
#Outputs:
#Val. vs Train Error %: Two line graphs on one plot. Red line is the validation
#                       error percentage, black line is training error percentage
#Mean Log Loss: Two line graphs on one plot. Red line is mean log loss for validation
#               data and black line is mean log loss for training data
programPlot <- function( numIterVector, trainErrorVector, validationErrorVector,
                         trainMeanLogLossVector, validationMeanLogLossVector, weightMatrix )
{
  #get mins for error percent
  minTrainerror.x <- which.min(trainErrorVector)
  minTrainerror.y <- min(trainErrorVector)
  
  print(c("min value for error per train is at itteration:", minTrainerror.x))
  
  minValiderror.x <- which.min(validationErrorVector)
  minValiderror.y <- min(validationErrorVector)
  
  print(c("min value for error per validation is at iteration:", minValiderror.x))
  
  
  #plot the validation error with the training error on the same graph with respect
  #to the number of iterations in the gradient descent
  dataSetErrorPer.plot<-ggplot() +
    geom_line(mapping=aes(1:ncol(weightMatrix),trainErrorVector,color="train")) +
    labs(x="Number of Itertions", y="Error Percent", color = "set") +
    geom_line(mapping=aes(1:ncol(weightMatrix),validationErrorVector,color="validation")) +
    geom_point(aes(x=minTrainerror.x, y=minTrainerror.y), color = "black")+
    geom_point(aes(x=minValiderror.x, y=minValiderror.y), color = "red")+
    scale_color_manual(values=c(train="black", validation="red"))
  
  print(dataSetErrorPer.plot)
  
  #get mins for mean log loss
  minMLLtrain.x<- which.min(trainMeanLogLossVector)
  minMLLtrain.y<- min(trainMeanLogLossVector)
  
  print(c("min value for mll train is at iteration:", minMLLtrain.x))
  
  minMLLvalid.x<- which.min(validationMeanLogLossVector)
  minMLLvalid.y<- min(validationMeanLogLossVector)
  
  print(c("min value for mll validation is at iteration:", minMLLvalid.x))
  
  #plot the mean log loss from validation set and training with respect
  #to the number of iterations in the gradient descent
  dataSetMLL.plot<-ggplot() +
    geom_line(mapping=aes(1:ncol(weightMatrix),trainMeanLogLossVector,color="train")) +
    labs(x="Number of Itertions", y="Mean Log Loss", color = "set") +
    geom_line(mapping=aes(1:ncol(weightMatrix),validationMeanLogLossVector,color="validation")) +
    geom_point(aes(x=minMLLtrain.x, y=minMLLtrain.y), color = "black")+
    geom_point(aes(x=minMLLvalid.x, y=minMLLvalid.y), color = "red")+
    scale_color_manual(values=c(train="black", validation="red"))
  
  print(dataSetMLL.plot)
  
  # #plot error percent
  # plot(numIterVector, trainErrorVector, type = 'l', col = "black", lwd = 3, xlab = "Number of Iterations", ylab = "Error Percent", ylim = range(0,.8))
  # lines(numIterVector, validationErrorVector, col = "red", lwd = 3)
  # title("Val. vs Train Error %")
  # legend(300,.35,c("Train","Validation"),lwd=c(3,3),col = c("black","red"), y.intersp=1.5)
  # points(which.min(validationErrorVector), min(validationErrorVector),cex = 2, col = 'red',pch=19)
  # points(which.min(trainErrorVector), min(trainErrorVector), cex = 2, col = 'black',pch=19)
  # 
  # #plot log loss
  # plot(numIterVector, trainMeanLogLossVector, type = 'l', col = "black", lwd = 3, xlab = "Number of Iterations", ylab = "Error Percent", ylim = range(0,.8))
  # lines(numIterVector, validationMeanLogLossVector, col = "red", lwd = 3)
  # title("Mean Log Loss")
  # legend("topright",c("Train","Validation"),lwd=c(3,3),col = c("black","red"), y.intersp=1.5)
  # points(which.min(validationMeanLogLossVector), min(validationMeanLogLossVector),cex = 2, col = 'red',pch=19)
  # points(which.min(trainMeanLogLossVector), min(trainMeanLogLossVector), cex = 2, col = 'black',pch=19)
  
}

#Function Name: clean
#Algorithm: Clean will "clean" our data set. It will remove
#           any columns with a standard deviation of zero,
#           which will break our data set if we run the
#           scale function on it.
clean <- function(X)
{
  #initialize column
  column = 1
  #go through every column in the X matrix
  while(column <= ncol(X))
  {
    #if sd of column is 0, delete it and reset column to 1
    if(sd(X[,column]) == 0)
    {
      #X <- X[,(-1 * column)] will return a matrix with every column except
      #the column'th column
      X <- X[,(-1 * column)]
      column = 1
    }
    #if sd of column is nonzero, then iterate and go to next column
    else
    {
      column = column + 1
    }
  }
  #return clean matrix
  X
}

#Function Name: predictY
#Algorithm: forms a y.hat vector that represents the predicted y value
#           given a output y vector, X data matrix, and predWeight predicted
#           weight.
#Inputs:
#y: data output
#X: data matrix
#predWeight: best set of predicted weights
#p: p value for which we classify a y as a 1, standard is 0.5
#Outputs:
#y.hat: vector of predicted values for y
predictY <- function( y, X, predWeight, p = 0.5 )
{
  #calculate y.hat using a combination sigmoid and round functions
  y.hat <- round(sigmoid(X %*% predWeight) + (.5-p))
  
  #return y.hat
  y.hat
}