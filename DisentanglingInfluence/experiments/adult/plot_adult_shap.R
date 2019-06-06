library(ggplot2)
setwd("~/Desktop/Drive/Research/MLResearch/DisentanglingInfluence/DisentanglingInfluence/experiments/adult")

adult_train = read.csv("../../data/adult/preprocessed_adult.train.csv")
adult_train = adult_train[0:5000,] # only calculated shap values for first 1000 points

shap = read.csv("adult_shap.csv")
adult_train$p_shap = shap$X0
adult_train$female = adult_train[["sex.Female"]]


figure = ggplot(adult_train, aes(x=p_shap, fill=factor(female))) + geom_histogram()
ggsave(filename="shap_values.png", figure)

