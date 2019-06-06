import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from DisentanglingInfluence.Influence.influence import DR_influence
from DisentanglingInfluence.Disentangling.disentangle import disentangle
from DisentanglingInfluence.utils import DataGenerator, padded_sigmoid
from DisentanglingInfluence.data.dsprites.reduce_dsprites import reduce_dsprites
from DisentanglingInfluence.Disentangling.model_factories.model_factory import ConvolutionalEncoder_DSprites
from DisentanglingInfluence.path import get_path
from DisentanglingInfluence.experiments.dsprite.dsprites_utils import show_images_grid
from DisentanglingInfluence.experiments.dsprite.dsprites_predictor import dsprites_predictor
from keras.models import load_model


def run_dsprite(train_steps=6000):
	# The number of data points to explain with SHAP values (the full dataset takes a while)
	N=3000

	# The size of the dataset to train the models on, 
	# and the size to report error metrics for (the full dataset is >700,000 images).
	# Setting either to False will result in using all instances for that case.
	n_train_instances = False
	n_report_instances = 10000
	
	# the image resolution to use. Currently, only 16 is supported.
	resolution = 16
	
	# trains a predictor if set to False
	pretrained_predictor = True
	
	#######################################

	path = get_path()
	output_dir = path + "outputs/dsprites_test/"
	
	filename = path + "data/dsprites/reduced_dsprites_{}.npz".format(resolution)
	if not os.path.isfile(filename):
		reduce_dsprites(new_dimension=resolution)
		print('Finished generating the reduced dsprites dataset.')
	
	zip_package = np.load(filename, mmap_mode="r", encoding="bytes")
	images = zip_package["imgs"]
	latents_classes = zip_package["latents_classes"][:,1:]
	latents_values = zip_package["latents_values"][:,1:]
	n_instances, x_dim, y_dim = images.shape
	n_feats = x_dim * y_dim
	
	if not n_train_instances:
		n_train_instances = n_instances
	if not n_report_instances:
		n_report_instances = n_instances
	
	# subsample, shuffle and flatten images
	keep_idxs = np.random.choice(n_instances, n_train_instances, replace=False)
	pd.DataFrame.from_dict({"data_indices":keep_idxs}).to_csv(output_dir + "data_indices.csv", index=False)
	images = images[keep_idxs]
	latents_values = latents_values[keep_idxs]
	
	shapeIsHeart = (latents_values[:,0] == 2)
	shapeIsHeart = shapeIsHeart.astype(int)
	latents_values[:,0] = shapeIsHeart  # replace shape with (shape == Heart)
	latents_values[:,2] = latents_values[:,2] / (2 * np.pi) # scale orientation to [0,1]
	
	latents_classes = latents_classes[keep_idxs]
	flat_images = images.reshape(n_train_instances, x_dim * y_dim)
		
	# read and parse the input data
	feature_names = ["shapeIsHeart", "scale", "orientation", "xPos", "yPos"]
	feature_types = ["binary", "continuous", "continuous", "continuous", "continuous"]
	label_name = "shapeIsHeart"
	
	if pretrained_predictor:
		clf = load_model(path + "experiments/dsprite/dsprites_predictor.h5")
	else:
		# train a model to predict the shape
		clf = dsprites_predictor(features=flat_images, labels=np.expand_dims(shapeIsHeart, axis=1), epochs=5)
		clf.save(path + "experiments/dsprite/dsprites_predictor.h5")
	
	
	results = {"influence":{feat:None for feat in feature_names},
				"reconstruction_error":{feat:None for feat in feature_names},
				"prediction_error":{feat:None for feat in feature_names},
				"discriminator_error":{feat:None for feat in feature_names}
				}
	
	# calculates the mean squared error
	mse = lambda x, y: np.mean((x-y)**2)
	
	for i, protected_feat in enumerate(feature_names):
		print("Calculating Influence for {}...".format(protected_feat))
		features = flat_images
		protected = latents_values[:,[feature_names.index(protected_feat)]]
		labels = latents_values[:,0] # label is the shape
		gen = DataGenerator([features, protected], batch_size=100)
		protected_is_categorical = (feature_types[i] == "binary")
		
		# train the models to disentangle the data
		FullModel, Enc, Dec, Disc, AutoEncoder = disentangle(data_generator=gen,
													learning_rate=0.05,
													latent_dim=6, 
													disc_weight=1,
													n_feats=n_feats, n_protected=1,
													train_steps=train_steps, enc_layer_sizes=[256, 32],
													dec_layer_sizes=[256], disc_layer_sizes=[64],
													dec_final_activ="sigmoid", 
													disc_final_activ="sigmoid",
													output_dir=output_dir,
													save_models=True,
													show_train_history=True,
													categorical_protected_feature=protected_is_categorical)
	
		sample_idxs = np.random.choice(list(range(flat_images.shape[0])), 25)
		sample = flat_images[sample_idxs]
		sample_protected = protected[sample_idxs]
		reconstructed = AutoEncoder.predict([sample, sample_protected])
		sample = sample.reshape(25, resolution, resolution)
		reconstructed = reconstructed.reshape(25, resolution, resolution)
	
		show_images_grid(sample)
		plt.savefig(output_dir + "original_imgs.png")
		plt.clf()
		show_images_grid(reconstructed)
		plt.savefig(output_dir + "reconstructed_imgs.png")
	
		trunc_features, trunc_protected = features[:n_report_instances], protected[:n_report_instances]
	
		# generate all the various representations
		unprotected = Enc.predict(trunc_features)  # latent representation
		dis_rep = [unprotected, trunc_protected] # full disentangled representation
		autoencoded = Dec.predict(dis_rep) # reconstructed original features
		phat = Disc.predict(unprotected) # revealed protected information
		preds = clf.predict(trunc_features) # model predictions on original data
		reconstructed_preds = clf.predict(autoencoded) # model predictions on reconstructed data
	
		# error metrics
		results["reconstruction_error"][protected_feat] = trunc_features - autoencoded
		results["prediction_error"][protected_feat] = preds.flatten() - reconstructed_preds.flatten()
		results["discriminator_error"][protected_feat] = phat.flatten() - trunc_protected.flatten()
		
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
	reconstruction_errors = {feat:pd.DataFrame(results["reconstruction_error"][feat]) for feat in feature_names}
	prediction_errors = pd.DataFrame.from_dict(results["prediction_error"])
	discriminator_mses = pd.DataFrame.from_dict(results["discriminator_error"])
	
	
	influences.to_csv(output_dir + "shap_values.csv", index=False)
	for feat in feature_names:
		recon_error_df = reconstruction_errors[feat]
		recon_error_df.to_csv(output_dir + "reconstruction_error_{}.csv".format(feat), index=False)
	prediction_errors.to_csv(output_dir + "prediction_error.csv", index=False)
	discriminator_mses.to_csv(output_dir + "discriminator_error.csv", index=False)

