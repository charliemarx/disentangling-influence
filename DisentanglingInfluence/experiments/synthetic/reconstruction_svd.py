from scipy.linalg import svd
import pandas as pd

def reconstruction_svd():
	filenames = [
	"results_SAVE3/reconstruction_error_x.csv",
	"results_SAVE3/reconstruction_error_x2.csv",
	"results_SAVE3/reconstruction_error_xSquared.csv",
	"results_SAVE3/reconstruction_error_y.csv",
	"results_SAVE3/reconstruction_error_y2.csv",
	"results_SAVE3/reconstruction_error_ySquared.csv",
	"results_SAVE3/reconstruction_error_z.csv",
	"results_SAVE3/reconstruction_error_z2.csv",
	"results_SAVE3/reconstruction_error_zSquared.csv"]

	for fname in filenames:
		recon_errors_df = pd.read_csv(fname)
		feature_names = recon_errors_df.columns
		recon_errors_matrix = recon_errors_df.values
		print(svd(recon_errors_matrix)[1])


reconstruction_svd()
	