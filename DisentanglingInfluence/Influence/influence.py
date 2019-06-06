
import keras.backend as K
import numpy as np
from DisentanglingInfluence.Influence.gradient import GradientExplainer

from keras.models import Model, Sequential 
from keras.utils import plot_model
from keras.layers import Concatenate, Input


def map2output(model, x):
	return model.predict(x)

def explainer(model, data, explain):
	"""
	model : The model to calculate the influence on
	data : The full dataset, as a tuple (X,y)
	explain : The subset of the X to explain the influence on
	"""
	
	X, y = data
	e = GradientExplainer(model, X, local_smoothing=0)
	shap_values, classes = e.shap_values(explain, ranked_outputs=1)
	return(shap_values)

def DR_influence(decoder, black_box, disentangled_reps, labels, explain=None, skip_disentangling=False):
	# explain the entire dataset if not specified
	if explain is None:
		explain = disentangled_reps

	##### compose the decoder and the model we are auditing
	if decoder:
		decoder_input_dims = [rep.shape[1:] for rep in disentangled_reps]
		decoder_inputs = [Input(shape=dim_i) for dim_i in decoder_input_dims]
		x_hat = decoder(decoder_inputs)
		y_hat = black_box(x_hat)
		composed  = Model(decoder_inputs, y_hat)

	else:
		composed = black_box

	# find the influence of each feature in the disentangled representation
	shap_values = explainer(composed,
					 (disentangled_reps, labels), 
					 explain)
	return(shap_values)


