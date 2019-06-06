import sys

sys.path.append('../..')

from DisentanglingInfluence.experiments.dsprite.run_dsprite import run_dsprite
from DisentanglingInfluence.experiments.dsprite.plot_shap_vals import plot_shap_vals
from DisentanglingInfluence.path import get_path

'''
runs the synthetic experiment. Results are saved in the outputs/synthetic_test directory. 
'''

path = get_path()
output_dir = path + "outputs/dsprites_test/"

run_dsprite(train_steps=6000)
plot_shap_vals(shapvals_filepath=output_dir + "shap_values.csv", 
	data_filepath=path + "data/dsprites/reduced_dsprites_16.npz",
	data_indices_filepath=output_dir + "data_indices.csv",
	indirect_output_filepath=output_dir + "indirect_influence_distributions.png",
	n_instances=3000)

