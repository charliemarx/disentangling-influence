from keras.activations import sigmoid

def padded_sigmoid(x):
	return(1.2*sigmoid(x)-0.1)

def stretched_sigmoid(x):
	return(2*sigmoid(x))

