library(BBmisc)
library(onehot)

# maps specified rare values to a default value
simplify = function(val, keep_vals, default) {
  if (val %in% keep_vals) {return(val)}
  else {return(default)}
} 
simplify = Vectorize(simplify, vectorize.args=c("val")) # vectorize the function

# returns the values in a vector which appear at least N times
countPlusN = function(vector, N) {
  plusNLevels = names(which(table(vector) > N))
  return(plusNLevels)
}

# returns the preprocessed data
preprocess = function(train_filename, test_filename, num_cols, cat_cols, drop_cols=FALSE, minCount=1000) {
  # read the data
  train = read.csv(train_filename)
  test = read.csv(test_filename)
  
  # create label for train vs test set
  train$set = factor(rep("train", dim(train)[1]))
  test$set = factor(rep("test", dim(test)[1]))
  
  # print dimensions
  cat("Train Shape:", dim(train), "\n")
  cat("Test Shape:", dim(test), "\n")
  
  # combine the data into one dataframe
  all_data = rbind(train, test)
  
  # drop unwanted features
  all_data[ ,drop_cols] <- list(NULL)
  for (drp in drop_cols) {
    cat("Dropping Unwanted Feature:", drp, "\n")
  }
  
  # normalize numerical features
  for (col in num_cols) {
    all_data[col] = normalize(all_data[col])
    cat("Normalizing:", col, "\n")
  }
  
  # bin rare values for categorical features
  for (col in cat_cols) {
    col_data = as.character(all_data[[col]])
    all_levels = levels(factor(col_data))
    common_levels = countPlusN(col_data, N=minCount)
    if (!setequal(common_levels, all_levels)) {
      cat("Simplifying:", col, "\n")
      all_data[col] = simplify(col_data,
                          keep_vals=common_levels, 
                          default=c("RareValue"))
    }
  }

  # onehot encode categorical features
  encoder = onehot(all_data, stringsAsFactors=TRUE, max_levels=99)
  
  # split into train and test sets
  train = all_data[all_data$set == "train",]
  test = all_data[all_data$set == "test",]
  
  # onehot encode each set
  prep_train = as.data.frame(predict(encoder, train))
  prep_test = as.data.frame(predict(encoder, test))
  
  # remove train vs. test indicators
  prep_train = prep_train[,setdiff(names(prep_train), c("set=train", "set=test"))]
  prep_test = prep_test[,setdiff(names(prep_test), c("set=train", "set=test"))]
  return(list(prep_train, prep_test))
}




