import pandas as pd
import numpy as np
from keras.layers import Lambda, Input, Concatenate, Dense, merge
from keras.models import Model
from keras import backend as K


def fixed_disentangler(protected_feature):
	feature_names = ["x","x2","xSquared","y","y2","ySquared","z","z2","zSquared"]
	n_features = len(feature_names)

	feat_input = Input(shape=(n_features,))
	protected_input = Input(shape=(1,))
	latent_input = Input(shape=(2,))

	latent = Dense(2) (feat_input)

	latent_1 = Lambda(lambda x: x[:,0:1], name="latent_1") (latent_input)
	latent_2 = Lambda(lambda x: x[:,1:2], name="latent_2") (latent_input)

	if "2" in protected_feature:
		protected_base = Lambda(lambda x: x / 2.) (protected_input)
	elif "Squared" in protected_feature:
		protected_base = Lambda(lambda x: K.sqrt(x)) (protected_input)
	else: 
		protected_base = protected_input

	# concat_layer has output [x,y,z]
	if "x" in protected_feature: # y z x
		encoder_weights = np.array([[int(feat == "y"),int(feat == "z")] for feat in feature_names])
		x = protected_base
		y = latent_1
		z = latent_2

	elif "y" in protected_feature: # x z y
		encoder_weights = np.array([[int(feat == "x"),int(feat == "z")] for feat in feature_names])	
		x = latent_1 
		y = protected_base
		z = latent_2

	elif "z" in protected_feature: # x y z
		encoder_weights = np.array([[int(feat == "x"),int(feat == "y")] for feat in feature_names])	
		x = latent_1 
		y = latent_2 
		z = protected_base

	else:
		raise(ValueError, "Error in protected feature name. Expected 'x', 'y', or 'z' to be in feature name.")


	square_layer = Lambda(lambda x: x**2)
	double_layer = Lambda(lambda x: 2*x)
	identity_layer = Lambda(lambda x: x)

	x_recon = Concatenate() ([identity_layer(x), double_layer(x), square_layer(x)])
	y_recon = Concatenate() ([identity_layer(y), double_layer(y), square_layer(y)])
	z_recon = Concatenate() ([identity_layer(z), double_layer(z), square_layer(z)])
	reconstructed = Concatenate() ([x_recon, y_recon, z_recon])

	Encoder = Model(feat_input, latent)
	Decoder = Model([latent_input, protected_input], reconstructed)
	Discriminator = Model(latent_input, Dense(1) (latent_input))

	Encoder.set_weights([encoder_weights])

	return(Encoder, Decoder, Discriminator)


def test_fixed_disentangler():
	enc, dec, disc = fixed_disentangler("zSquared")
	encoder_pred = list(enc.predict([[[1,2,3,4,5,6,7,8,9]]])[0]) # [[[x, x2, xSquared, y, etc.]]]
	decoder_pred = list(dec.predict([[[1,2]],[[9]]])[0])  # [[[x,y]], [[z^2]]]
	print(decoder_pred)
	print("Encoder working correctly? --", encoder_pred == [1,4])
	print("decoder working correctly? --", decoder_pred == [1,2,1,2,4,4,3,6,9])

if __name__ == "__main__":
	test_fixed_disentangler()



