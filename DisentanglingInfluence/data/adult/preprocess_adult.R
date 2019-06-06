setwd("~/Desktop/Drive/Research/MLResearch/DisentanglingInfluence/DisentanglingInfluence/data/adult")
source("../preprocess.R")

prep_adult = function(return_results=FALSE, write_results=TRUE) {

  # sort features by type of data
  dropped = c("fnlwgt")
  numerical = c("age", "capital.gain", "capital.loss", "hours.per.week", "education.num")
  categorical = c("workclass", "education", 
                  "marital.status", "occupation", 
                  "relationship", "race", "sex", "native.country")
  
  # preprocess the data
  prep = preprocess(train_filename="adult.train.csv",
                    test_filename="adult.test.csv",
                    num_cols=numerical,
                    cat_cols=categorical,
                    drop_cols=dropped)


  redundant_features = c("sex=Female", "income.per.year=<=50K")
  # unpack the train and test dataframes
  prep_train = prep[[1]][ , !(names(prep[[1]]) %in% redundant_features)]
  prep_test = prep[[2]][ , !(names(prep[[2]]) %in% redundant_features)]
  
  # save the preprocessed data
  if (write_results == TRUE) {
    write.csv(prep_train, "preprocessed_adult.train.csv", row.names=FALSE)
    write.csv(prep_test, "preprocessed_adult.test.csv", row.names=FALSE)
  }
  
  cat("Completed.\n")
  if (return_results) {return(prep)}
}

adult_processed = prep_adult(return_results=TRUE, write_results=FALSE)
adult_train = adult_processed[[1]]
adult_test = adult_processed[[2]]