source("GradientDescentHelper.R")

main <- function()
{
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
  X <- import_data[,1:57]
  X <- scale(X)
  y <- import_data[,58]
  
  #separate data using classificationList as needed
  X_train <- X[classificationList == 0,]
  X_validation <- X[classificationList == 1,]
  X_test <- X[classificationList == 2,]
  
  #separate y using 
  y_train <- y[classificationList == 0]
  y_validation <- y[classificationList == 1]
  y_test <- y[classificationList == 2]
  
  #store weightMatrix
  weightMatrix <- GradientDescent( X_train, y_train, stepSize, maxIterations )
  
  #initialize vector variables
  trainErrorVector = NA
  validationErrorVector = NA
  trainLogLossVector = NA
  validationMeanLogLossVector = NA 
  for(i in 1:(maxIterations+1))
  {
    trainErrorVector[i] <- 1 - findAccuracy( X_train, y_train, weightMatrix[,i])
    validationErrorVector[i] <- 1 - findAccuracy( X_validation, y_validation, weightMatrix[,i] )
    trainMeanLogLossVector[i] <- calculateMeanLogLoss( X_train, y_train, weightMatrix[,i] )
    validationMeanLogLossVector[i] <- calculateMeanLogLoss( X_validation, y_validation, weightMatrix[,i] )
  }
  
  #plot the validation error with the training error on the same graph with respect
  #to the number of iterations in the gradient descent
  ggplot() +
  geom_smooth(mapping=aes(1:ncol(weightMatrix),trainErrorVector),color = 'red') +
  labs(x="Number of Iteartions") +
  geom_smooth(mapping=aes(1:ncol(weightMatrix),validationErrorVector)) +
  labs(x="Number of Iterations")
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
  X <- import_data[,1:ncol(import_data)-1]
  X <- scale(X)
  y <- import_data[,ncol(import_data)-1]
  
  #separate data using classificationList as needed
  X_train <- X[classificationList == 0,]
  X_validation <- X[classificationList == 1,]
  X_test <- X[classificationList == 2,]
  
  #separate y using 
  y_train <- y[classificationList == 0]
  y_validation <- y[classificationList == 1]
  y_test <- y[classificationList == 2]
  
  #store weightMatrix
  weightMatrix <- GradientDescent( X_train, y_train, stepSize, maxIterations )
  
  #initialize vector variables
  trainErrorVector = NA
  validationErrorVector = NA
  trainLogLossVector = NA
  validationMeanLogLossVector = NA 
  for(i in 1:(maxIterations+1))
  {
    trainErrorVector[i] <- 1 - findAccuracy( X_train, y_train, weightMatrix[,i])
    validationErrorVector[i] <- 1 - findAccuracy( X_validation, y_validation, weightMatrix[,i] )
    trainMeanLogLossVector[i] <- calculateMeanLogLoss( X_train, y_train, weightMatrix[,i] )
    validationMeanLogLossVector[i] <- calculateMeanLogLoss( X_validation, y_validation, weightMatrix[,i] )
  }
  
  #############################################################################
  #############################################################################
  #                             Third Data Set                                #
  #############################################################################
  #############################################################################
  
  #Return a weightMatrix
  weightMatrix
}

#run this big boy
weightMatrix <- main()
###############################################################################
###############################################################################
#                                          Testing                            #
###############################################################################
###############################################################################
source("GradientDescentHelper.R")
###############################################################################
#                                       Stuff                                 #
###############################################################################
#import data as a list
import_data <- fread('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data')
#convert list to matrix
import_data <- matrix( unlist(import_data), nrow(import_data), ncol(import_data) )
print("done downloading")

#randomized list of 0's 1's and 2's
classificationList <- createClassificationList(import_data)

#make X and Y vector/matrix
X <- import_data[,1:57]
X <- scale(X)
y <- import_data[,58]

#separate data using classificationList as needed
X_train <- X[classificationList == 0,]
X_validation <- X[classificationList == 1,]
X_test <- X[classificationList == 2,]

#separate y using 
y_train <- y[classificationList == 0]
y_validation <- y[classificationList == 1]
y_test <- y[classificationList == 2]

weightMatrix <- GradientDescent( X_train, y_train, stepSize, maxIterations )
###############################################################################
#                                    End Stuff                                #
###############################################################################
trainErrorVector = NA
validationErrorVector = NA
trainMeanLogLossVector = NA
validationMeanLogLossVector = NA
for(i in 1:(maxIterations+1))
{
  trainErrorVector[i] <- 1 - findAccuracy(X_train,y_train,weightMatrix[,i])
  validationErrorVector[i] <- 1 - findAccuracy( X_validation, y_validation, weightMatrix[,i] )
  trainMeanLogLossVector[i] <- calculateMeanLogLoss(X_train, y_train, weightMatrix[,i])
  validationMeanLogLossVector[i] <- calculateMeanLogLoss(X_validation,y_validation,weightMatrix[,i])
}
plot(1:501, trainMeanLogLossVector)
plot(1:501, validationMeanLogLossVector)

#get mins for error percent
minTrainerror.x <- which.min(trainErrorVector)
minTrainerror.y <- min(trainErrorVector)

print("min value for errorper train is at itteration:", minValiderror.x)

minValiderror.x <- which.min(validationErrorVector)
minValiderror.y <- min(validationErrorVector)

print("min value for errorper validation is at iteration:", minValiderror.x)


#plot the validation error with the training error on the same graph with respect
#to the number of iterations in the gradient descent
thirdSetErrorPer.plot<-ggplot() +
  geom_line(mapping=aes(1:ncol(weightMatrix),trainErrorVector,color="train")) +
  labs(x="Number of Itertions", y="Error Percent") +
  geom_line(mapping=aes(1:ncol(weightMatrix),validationErrorVector,color="validation")) +
  geom_point(aes(x=minTrainerror.x, y=minTrainerror.y, color = "min"))+
  geom_point(aes(x=minValiderror.x, y=minValiderror.y,color = "min"))+
  scale_color_manual(values=c(train="black", validation="red", min ="blue"))

print(thirdSetErrorPer.plot)

#get mins for mean log loss
minMLLtrain.x<- which.min(trainMeanLogLossVector)
minMLLtrain.y<- min(trainMeanLogLossVector)

print("min value for mll train is at iteration:", minMLLvalid.x)

minMLLvalid.x<- which.min(validationMeanLogLossVector)
minMLLvalid.y<- min(validationMeanLogLossVector)

print("min value for mll validation is at iteration:", minMLLvalid.x)

#plot the mean log loss from validation set and training with respect
#to the number of iterations in the gradient descent
thirdSetMLL.plot<-ggplot() +
  geom_line(mapping=aes(1:ncol(weightMatrix),trainMeanLogLossVector,color="train")) +
  labs(x="Number of Itertions", y="Mean Log Loss") +
  geom_line(mapping=aes(1:ncol(weightMatrix),validationMeanLogLossVector,color="validation")) +
  geom_point(aes(x=minMLLtrain.x, y=minMLLtrain.y, color = "min"))+
  geom_point(aes(x=minMLLvalid.x, y=minMLLvalid.y,color = "min"))+
  scale_color_manual(values=c(train="black", validation="red", min ="blue"))

print(thirdSetMLL.plot)

