from matplotlib import rc
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from keras.models import load_model


def plot_shap_vals(
	shapvals_filepath, 
	data_filepath, 
	data_indices_filepath,
	indirect_output_filepath, 
	n_instances):

	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	## for Palatino and other serif fonts use:
	rc('font',**{'family':'serif','serif':['Palatino']})
	rc('text', usetex=False)
	
	shap.initjs()

	zip_package = np.load(data_filepath, mmap_mode="r", encoding="bytes")
	latents_values = zip_package["latents_values"][:,1:]
	idxs = pd.read_csv(data_indices_filepath)["data_indices"]

	# match indices with training
	latents_values = latents_values[idxs]
	shapeIsHeart = (latents_values[:,0] == 1)
	shapeIsHeart = shapeIsHeart.astype(int)
	latents_values[:,0] = shapeIsHeart  # replaces shape feature with (shape == Heart)
		
	shap_vals_df = pd.read_csv(shapvals_filepath)
	feat_names = ["shape", "scale", "orient.", "x pos.", "y pos."]
	feats = latents_values[0:n_instances]
	shap_vals = shap_vals_df[0:n_instances].values
	
	shap.summary_plot(shap_vals, feats, show=False, plot_type="dot", sort=False, feature_names= feat_names)
	
	plt.savefig(indirect_output_filepath)
	plt.clf()
	




