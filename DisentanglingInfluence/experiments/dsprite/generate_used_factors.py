import numpy as np
import pandas as pd

def generate_used_factors(indices_filename, data_filename, out_filename, factor_names):
	zip_package = np.load(data_filename, mmap_mode="r", encoding="bytes")
	latents_values = zip_package["latents_values"][:,1:]
	shapeIsHeart = (latents_values[:,0] == 2)
	shapeIsHeart = shapeIsHeart.astype(int)
	latents_values[:,0] = shapeIsHeart  # replace shape with (shape == Heart)
	latents_values[:,2] = latents_values[:,2] / (2 * np.pi) # scale orientation to [0,1]

	indices = pd.read_csv(indices_filename).values 
	used_factors = pd.DataFrame(latents_values[indices].squeeze())
	used_factors.columns = factor_names
	used_factors.to_csv(out_filename, index=False)
	print("Completed.")


factor_names = ["shapeIsHeart", "scale", "orientation", "xPos", "yPos"]

generate_used_factors(indices_filename="results_2/data_indices.csv",
					data_filename="../../data/dsprites/reduced_dsprites_16.npz",
					out_filename="results_2/data_factors.csv",
					factor_names=factor_names)

