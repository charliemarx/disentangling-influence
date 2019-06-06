import pandas as pd
from DisentanglingInfluence.utils import *
from DisentanglingInfluence.path import get_path
from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l1


def sum_predictor():

	# get the absolute path to the package
	path = get_path()

	# read and parse the input data
	data = pd.read_csv(path + "data/synthetic/sum_synthetic.csv")
	data = data.sample(frac=1).reset_index(drop=True) # shuffle instances

	feature_names = ["x","x2","xSquared","y","y2","ySquared","z","z2","zSquared"]
	label_name = "xPy_Label"

	
	features = data[feature_names].values
	labels = data[label_name].values
	
	n_instances, n_feats = features.shape
	

	r_input = Input(shape=(n_feats,))
	layer = Dense(1, name="y_hat", kernel_regularizer=l1(0.001)) (r_input)

	model = Model(inputs=[r_input], outputs=[layer])

# 	model.compile(optimizer="rmsprop", loss="mse")
	model.compile(optimizer="sgd", loss="mse")
	model.fit(features, labels, epochs=15, batch_size=16)
	return(model)

def fixed_predictor():

	feature_names = ["x","x2","xSquared","y","y2","ySquared","z","z2","zSquared"]
	n_feats = len(feature_names)

	r_input = Input(shape=(n_feats,))
	layer = Dense(1, name="y_hat") (r_input)
	model = Model(inputs=[r_input], outputs=[layer])

	model_weights = np.array([[1] if feat in ["x", "y"] else [0] for feat in feature_names])

	model.set_weights([model_weights])
	return(model)

if __name__ == "__main__":
	model = fixed_predictor()
	print("Fixed model predicting as expected? --", list(model.predict([[[11,22,33,44,55,66,77,88,99]]])) == [[55]])

	




