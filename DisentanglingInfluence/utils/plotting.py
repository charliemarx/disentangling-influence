import matplotlib.pyplot as plt

def index_plot(vector, filename, x_lab=None, y_lab=None, title=None):
	y = vector
	x = range(len(y))
	plt.plot(x, y)
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	plt.title(title)
	plt.savefig(filename)
	plt.clf()