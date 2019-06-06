import sys

sys.path.append('../..')

from DisentanglingInfluence.experiments.synthetic.run_synthetic import run_synthetic
from DisentanglingInfluence.experiments.synthetic.plot_shap_vals import plot_shap_vals
from DisentanglingInfluence.path import get_path

'''
runs the synthetic experiment. Results are saved in the outputs/synthetic_test directory. 
'''

path = get_path()

run_synthetic(train_steps=8000)
plot_shap_vals(shapvals_filepath=path + "outputs/synthetic_test/shap_values.csv", 
	data_filepath=path + "data/synthetic/sum_synthetic.csv",
	indirect_output_filepath=path + "outputs/synthetic_test/indirect_influence_distributions.png",
	direct_output_filepath=path + "outputs/synthetic_test/direct_influence_distributions.png",
	predictor_filepath=path + "experiments/synthetic/models/sum_predictor.h5",
	n_instances=3000)

