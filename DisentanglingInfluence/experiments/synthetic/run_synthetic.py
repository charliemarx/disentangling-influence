import pandas as pd
import numpy as np
from DisentanglingInfluence.Influence.influence import DR_influence
from DisentanglingInfluence.Disentangling.disentangle import disentangle
from DisentanglingInfluence.utils import DataGenerator, stretched_sigmoid
from DisentanglingInfluence.experiments.synthetic.predict_sum import sum_predictor, fixed_predictor
from DisentanglingInfluence.experiments.synthetic.fixed_disentangle import fixed_disentangler
from DisentanglingInfluence.path import get_path

# The number of data points to explain with SHAP values (the full dataset takes a while)
N=3000

USE_FIXED_PREDICTOR = True
USE_FIXED_DISENTANGLER = False

path = get_path()
output_dir = path + "outputs/synthetic_test/"

def run_synthetic(train_steps=10000):
	# read and parse the input data
	data = pd.read_csv(path + "data/synthetic/sum_synthetic.csv")
	feature_names = ["x","x2","xSquared","y","y2","ySquared","z","z2","zSquared"]
	label_name = "xPy_Label"
	
	# train a regressor for the label (x+y) from the features
	if USE_FIXED_PREDICTOR:
		clf = fixed_predictor()	
	else:
		clf = sum_predictor()

	clf.save(path + "experiments/synthetic/models/sum_predictor.h5")
	
	
	results = {"influence":{feat:None for feat in feature_names},
				"reconstruction_error":{feat:None for feat in feature_names},
				"prediction_error":{feat:None for feat in feature_names},
				"discriminator_error":{feat:None for feat in feature_names}
				}
	
	# calculates the mean squared error
	mse = lambda x, y: np.mean((x-y)**2)
	
	for protected_feat in feature_names:
		print("Calculating Influence for {}...".format(protected_feat))
		# unprotected_names = [feat for feat in feature_names if feat is not protected_feat]
		features = data[feature_names].values
		protected = data[[protected_feat]].values
		labels = data["xPy_Label"].values
		n_instances, n_feats = features.shape
		gen = DataGenerator([features, protected], batch_size=16)
		
	
		if USE_FIXED_DISENTANGLER:
			Enc, Dec, Disc = fixed_disentangler(protected_feature=protected_feat)
	
		else:
			if "2" in protected_feat:
				disc_final_activ = stretched_sigmoid
			else:
				disc_final_activ = "sigmoid"
			# train the models to disentangle the data
			FullModel, Enc, Dec, Disc, AutoEncoder = disentangle(data_generator=gen,
												latent_dim=4, 
												disc_weight=0.5,
												n_feats=n_feats, n_protected=1,
												train_steps=train_steps, enc_layer_sizes=[10,10],
												dec_layer_sizes=[10,10], disc_layer_sizes=[10,10],
												dec_final_activ=stretched_sigmoid, 
												disc_final_activ=disc_final_activ,
												output_dir=output_dir)
	
		
		# generate all the various representations
		unprotected = Enc.predict(features)  # latent representation
		dis_rep = [unprotected, protected] # full disentangled representation
		autoencoded = Dec.predict(dis_rep) # reconstructed original features
		phat = Disc.predict(unprotected) # revealed protected information
		preds = clf.predict(features) # model predictions on original data
		reconstructed_preds = clf.predict(autoencoded) # model predictions on reconstructed data
	
		# error metrics
		results["reconstruction_error"][protected_feat] = features - autoencoded
		results["prediction_error"][protected_feat] = preds.flatten() - reconstructed_preds.flatten()
		results["discriminator_error"][protected_feat] = protected.flatten() - phat.flatten()
	
		
		# choose a subset of the data to explain the influence on
		explain = [rep[0:N] for rep in dis_rep]
		
		# calculate the influence!
		influence = DR_influence(decoder=Dec, 
			black_box=clf, 
			disentangled_reps=dis_rep, 
			labels=labels, 
			explain=explain)
	
		results["influence"][protected_feat] = influence[0][1].flatten()
	
	# report results (influence and error metrics)
	influences = pd.DataFrame.from_dict(results["influence"])
	reconstruction_errors = {feat:pd.DataFrame(results["reconstruction_error"][feat], columns=feature_names) for feat in feature_names}
	prediction_errors = pd.DataFrame.from_dict(results["prediction_error"])
	discriminator_mses = pd.DataFrame.from_dict(results["discriminator_error"])
	
	
	influences.to_csv(output_dir + "shap_values.csv", index=False)
	for feat in feature_names:
		recon_error_df = reconstruction_errors[feat]
		recon_error_df.to_csv(output_dir + "reconstruction_error_{}.csv".format(feat), index=False)
	prediction_errors.to_csv(output_dir + "prediction_error.csv", index=False)
	discriminator_mses.to_csv(output_dir + "discriminator_error.csv", index=False)

