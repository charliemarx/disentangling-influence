from matplotlib import rc
import pandas as pd
import shap
import matplotlib.pyplot as plt
from keras.models import load_model
import sys

#######

def plot_shap_vals(
	shapvals_filepath, 
	data_filepath, 
	indirect_output_filepath, 
	direct_output_filepath,
	predictor_filepath,
	n_instances):

	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	## for Palatino and other serif fonts use:
	rc('font',**{'family':'serif','serif':['Palatino']})

	# use TeX if the user has it installed...
	sent_tex_warning = False
	try:
		rc('text', usetex=True)
	except:
		rc('text', usetex=False)
		print("WARNING: Producing figure without latex since it is not installed.")
		sent_tex_warning = True
	
	shap.initjs()
		
	shap_vals_df = pd.read_csv(shapvals_filepath)
	feats_df = pd.read_csv(data_filepath)
	feat_names = ["x","x2","xSquared","y","y2","ySquared","z","z2","zSquared"]
	
	feats_df = feats_df[feat_names]
	shap_vals_df = shap_vals_df[feat_names]
	
	feats = feats_df[0:n_instances].values
	shap_vals = shap_vals_df[0:n_instances].values
	
	try:
		shap.summary_plot(shap_vals, feats, show=False, plot_type="dot", sort=False, feature_names= [r"$x$",r"$2x$",\
	                                                        r"$x^2$",r"$y$",r"$2y$", r"$y^2$",r"$c$",r"$2c$",r"$c^2$"])
	except:
		plt.clf()
		shap.summary_plot(shap_vals, feats, show=False, plot_type="dot", sort=False, 
			feature_names=["x","2x","x^2","y","2y", 
							"y^2","c","2c","c^2"])
		if not sent_tex_warning:
			print("WARNING: Producing figure without latex since it is not installed.")
			sent_tex_warning=True

	plt.savefig(indirect_output_filepath)
	plt.clf()


	# plot direct influence for baseline
	predictor = load_model(predictor_filepath)
	e = shap.GradientExplainer(predictor, feats, local_smoothing=0)
	shap_values, classes = e.shap_values(feats, ranked_outputs=1)

	# use TeX if the user has it installed...
	try:
		shap.summary_plot(shap_values[0], feats, show=False, plot_type="dot", sort=False, 
			feature_names= [r"$x$",r"$2x$",r"$x^2$",r"$y$",r"$2y$", 
							r"$y^2$",r"$c$",r"$2c$",r"$c^2$"])
	except:
		plt.clf()
		shap.summary_plot(shap_values[0], feats, show=False, plot_type="dot", sort=False, 
			feature_names=["x","2x","x^2","y","2y", 
							"y^2","c","2c","c^2"])
		if not sent_tex_warning:
			print("WARNING: Producing figure without latex since it is not installed.")

	plt.savefig(direct_output_filepath)





