import keras.backend as K
from keras.callbacks import Callback


def binary_BCR(y_true, y_pred):

	ones_like_true = K.cast(K.round(y_true), K.floatx())
	ones_like_pred = K.round(y_pred)

	# count correct predictions
	is_correct = K.cast(K.equal(ones_like_true, ones_like_pred), K.floatx())
	num_correct = K.sum(is_correct, axis=0)

	# count correct positives and negatives
	TP = K.sum(K.tf.multiply(is_correct, ones_like_true))
	TN = K.tf.subtract(num_correct, TP)

	# count total positive and negative
	total_pos = K.sum(ones_like_true, axis=0)
	total_neg = K.cast(K.shape(ones_like_true)[0], K.floatx()) - total_pos

	# compute each accuracy
	neg_acc = TN / total_neg
	pos_acc = TP / total_pos

	# create a mask for whether each accuracy is NaN
	rates = K.concatenate([pos_acc, neg_acc], axis=0)
	nans = K.cast(K.tf.is_nan(rates), "float32")
	not_nans = K.equal(nans, K.zeros_like(rates)) # invert the truth values

	# apply mask to get rid of NaNs
	masked = K.tf.boolean_mask(tensor=K.cast(rates, "float32"), mask=not_nans)

	# divide sum of accuracies by number of accuracies used
	bcr = K.sum(masked) / K.sum(K.cast(not_nans, "float32"))
	return(bcr) 
