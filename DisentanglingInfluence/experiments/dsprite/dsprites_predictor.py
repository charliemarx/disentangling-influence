
import pandas as pd
from DisentanglingInfluence.utils import *
from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l1



def dsprites_predictor(features, labels, val_split=0.2, epochs=5):

	
	n_instances, n_feats = features.shape
	"""
	if validate:
		train_cutoff = 0.8 * n_instances

		features = features[:train_cutoff]
		labels = labels[:train_cutoff]
		valid_features = features[train_cutoff:]
		valid_labels = labels[train_cutoff:]
	"""
	

	r_input = Input(shape=(n_feats,))
	layer = Dense(128, activation="relu") (r_input)
	layer = Dense(64, activation="relu") (layer)
	layer = Dense(32, activation="relu") (layer)
	layer = Dense(1, name="y_hat", activation="sigmoid") (layer)



	model = Model(inputs=[r_input], outputs=[layer])

# 	model.compile(optimizer="rmsprop", loss="mse")
	model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["acc", binary_BCR])
	model.fit(features, labels, epochs=epochs, batch_size=32, validation_split=val_split)
	return(model)


if __name__ == "__main__":
	model = dsprites_predictor()

	




