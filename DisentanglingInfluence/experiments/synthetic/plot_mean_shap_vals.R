library(latex2exp)
library(ggplot2)

feat_names = c(TeX("$X$"), TeX("$2X$"), TeX("$X^2$"),
          TeX("$Y$"), TeX("$2Y$"), TeX("$Y^2$"),
          TeX("$Z$"), TeX("$2Z$"), TeX("$Z^2$"))

data = read.csv("../../../data/synthetic/sum_synthetic.csv")
shap_values = read.csv("shap_values.csv")
print(names(shap_values))
shap_means = apply(shap_values, 2, absmean)
print(shap_means)
#df = data.frame(shap_means, shap_means)
#p = ggplot(df, aes(x=factor(shap_means), y=shap_means)) + geom_bar(stat="identity")
#p = p + geom_text(label=feat_names)
barplot(shap_means, names.arg=feat_names, xlab="Feature", ylab="Mean Absolute Influence", main="Indirect Influence Scores")