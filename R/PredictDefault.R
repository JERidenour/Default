library(DMwR)

# load and format data
rawData <- read.csv("Data/dataset (3).csv", sep=";", header=TRUE)
categorical <- c("default", "account_status", "account_worst_status_0_3m", 
                 "account_worst_status_12_24m", "account_worst_status_3_6m",
                 "account_worst_status_6_12m", "merchant_category", "merchant_group",
                 "has_paid", "name_in_email", "status_last_archived_0_24m", 
                 "status_2nd_last_archived_0_24m", "status_3rd_last_archived_0_24m",
                 "status_max_archived_0_6_months", "status_max_archived_0_12_months", 
                 "status_max_archived_0_24_months", "worst_status_active_inv") 
rawData[categorical] <- lapply(rawData[categorical] , factor) #format categorical variables

# split the prediction/model data
predictionData <- rawData[is.na(rawData$default),] #data we want predictions for
modelData <- rawData[!is.na(rawData$default),] #data for use in developing/testing the model

# split model data into training & test sets
set.seed(5) 
train <- sample(1:nrow(modelData), 4*nrow(modelData)/5) #(80/20) split
modelDataTrain <- modelData[train,]
modelDataTest <- modelData[-train,]

# use knn to fill missing data in test and prediction sets
modelDataTest <- modelDataTest[, -2] #remove default
predictionData <- predictionData[, -2]
knn.model.test <- knnImputation(modelDataTest)  #model missing values with knn (this takes ~7,8 min)
knn.model.prediction <- knnImputation(predictionData) 

# train the model
glm.model <- glm(default ~ 
                    account_worst_status_0_3m + 
                    account_worst_status_12_24m + 
                    age +  
                    avg_payment_span_0_12m +
                    avg_payment_span_0_3m +
                    num_active_div_by_paid_inv_0_12m +
                    status_max_archived_0_6_months + 
                    worst_status_active_inv, 
                    data = modelDataTrain, family=binomial)

# predict on the test data
testProbs <- predict(glm.model, newdata = knn.model.test, type="response")
testPred <- rep(0, nrow(knn.model.test))
testPred[testProbs>mean(testProbs)] <- 1 #predict default for probabilities greater than ...
#testPred[testProbs>0.003] <- 1

# see a confusion matrix of the results
table(testPred, modelData$default[-train])

# I get:
# testPred     0     1
#        0 16149   140
#        1  1598   109
# ==> test error rate: 140/(140+109) = 0.562249

# retrain using the full model data
glm.model <- glm(default ~ 
                   account_worst_status_0_3m + 
                   account_worst_status_12_24m + 
                   age +  
                   avg_payment_span_0_12m +
                   avg_payment_span_0_3m +
                   num_active_div_by_paid_inv_0_12m +
                   status_max_archived_0_6_months + 
                   worst_status_active_inv, 
                   data = modelData, family=binomial)

# assemble and print the final results
finalProbs <- predict(glm.model, newdata = knn.model.prediction, type="response")
finalResults <- data.frame(knn.model.prediction$uuid, finalProbs) 
names(finalResults) <- c("uuid", "pd")
write.table(finalResults, file = "Results.csv", sep=";", row.names = FALSE)
