import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from DisentanglingInfluence.path import get_path

"""

def adult_classifier_DL(epochs=5):
	# read and parse the input data
	data = pd.read_csv("../../data/adult/preprocessed_adult.train.csv")
	data = data.sample(frac=1).reset_index(drop=True) # shuffle instances
	non_feat_names = ["sex=Male", "sex=Female", "income.per.year=<=50K", "income.per.year=>50K"]
	feat_names = [col for col in data.columns if col not in non_feat_names]
	features = data[feat_names].values
	labels = data["income.per.year=>50K"].values
	
	n_instances, n_feats = features.shape

	#test=
	

	r_input = Input(shape=(n_feats,))
	layer = Dense(30, activation="tanh", input_dim=n_feats, name="clf_1") (r_input)
	layer = Dense(20, activation="tanh", input_dim=n_feats, name="clf_2") (layer)
	layer = Dense(10, activation="tanh", input_dim=n_feats, name="clf_3") (layer)
	layer = Dense(1, name="y_hat", activation="sigmoid") (layer)

	model = Model(inputs=[r_input], outputs=[layer])

	model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
	model.fit(features, labels, epochs=epochs, validation_data= batch_size=16)
	return(model)
"""

def adult_classifier(exclude_sex=False, epochs=4):

    # get absolute path to DisentanglingInfluence
    path = get_path()

	# read and parse the input data
    train = pd.read_csv(path + "data/adult/preprocessed_adult.train.csv")
    test = pd.read_csv(path + "data/adult/preprocessed_adult.test.csv")
    train = train.sample(frac=1).reset_index(drop=True) # shuffle instances

    if exclude_sex:
        non_feat_names = ["sex=Male", "income.per.year=>50K"]
    else:
        non_feat_names = ["income.per.year=>50K"]
    feat_names = [col for col in train.columns if col not in non_feat_names]
    features = train[feat_names].values
    labels = train["income.per.year=>50K"].values

    test_features = test[feat_names].values
    test_labels = test["income.per.year=>50K"].values

    # fit an LR model to the data
    # model = LogisticRegressionCV(random_state=0, solver='newton-cg',n_jobs=-1, cv=5).fit(features,labels)

    n_instances, n_feats = features.shape
    r_input = Input(shape=(n_feats,))
    layer = Dense(1, name="y_hat", activation="sigmoid")(r_input)

    model = Model(inputs=[r_input], outputs=[layer])
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["acc"])
    model.fit(features, labels, epochs=epochs)
    loss, acc = model.evaluate(x=test_features, y=test_labels)
    print("Test Loss: {} --- Test Accuracy: {}".format(loss, acc))

    return model


if __name__ == "__main__":
    model = adult_classifier()

