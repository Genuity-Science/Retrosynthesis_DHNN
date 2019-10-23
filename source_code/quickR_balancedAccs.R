library (caret)

#load list as vectors:
predResults <- unlist(read.table("PredictedLabelsForR.txt",header = FALSE))
trueResults <- unlist(read.table("TrueLabelsForR.txt", header = FALSE))

#get confusion matrix:
cm <- confusionMatrix(predResults,trueResults)

#get balanced accuracies:
byClass_table <-cm$byClass
balancedAccuracyValues <- byClass_table[,"Balanced Accuracy"]
averageBalancedAccuracy <- mean(balancedAccuracyValues, na.rm =T) 

write.table(balancedAccuracyValues, file="BalancedAccuraciesFromCaretR.csv",col.names=F,sep=',')