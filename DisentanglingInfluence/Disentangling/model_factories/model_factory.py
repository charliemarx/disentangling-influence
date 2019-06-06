import keras.backend as K
from keras.layers import Dense, Lambda, Concatenate, Conv2D, Flatten, MaxPooling2D, Reshape
from keras.regularizers import l2, l1

from DisentanglingInfluence.Disentangling.model_factories.abstract_model_factory import *

"""
Default model builders for the disentangling process.
Author: -----

"""

class Encoder(AbstractEncoder):
	# symbolic output for encoder
	def build(self, layer_sizes):
		layer = self.feat_input
		for l_size in layer_sizes:
			layer = Dense(l_size, activation="relu") (layer)
		layer = Dense(self.latent_dim, activation=None, kernel_regularizer=l1(0.0)) (layer)
		return(layer)

class ConvolutionalEncoder_DSprites(AbstractEncoder):
	# symbolic output for encoder
	def build(self, layer_sizes):
		layer = self.feat_input
		layer = Reshape((16, 16)) (layer)
		layer = Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(16, 16)) (layer)
		layer = Conv2D(64, kernel_size=(3,3), activation='relu') (layer)
		layer = MaxPooling2D(pool_size=(2, 2)) (layer)
		layer = Flatten() (layer)
		layer = Dense(self.latent_dim) (layer)


# Not ready, just threw some code together, won't run
class ConvolutionalDecoder_DSprites(AbstractDecoder):
	# symbolic output for decoder
	def build(self, layer_sizes, final_activation=None):
		layer = Concatenate() ([self.latent_input, self.protected_input])
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
		x = UpSampling2D((2, 2))(x)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((2, 2))(x)
		x = Conv2D(16, (3, 3), activation='relu')(x)
		x = UpSampling2D((2, 2))(x)
		decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
		layer = Dense(self.feat_dim, activation=final_activation, name="x_hat") (layer)
		return(layer)

class VariationalEncoder(AbstractEncoder):
	# symbolic output for encoder
	def build(self, layer_sizes):
		layer = Dense(layer_sizes[0], activation="relu") (self.feat_input)
		for l_size in layer_sizes[1:]:
			layer = Dense(l_size, activation="relu") (layer)
		mu = Dense(self.latent_dim, name="mu", activity_regularizer=l2(0.001)) (layer)
		log_sigma = Dense(self.latent_dim, name="log_sigma", activity_regularizer=l2(0.001)) (layer)
		z = Lambda(self.sampling, name="z") ([mu, log_sigma])
		return(z)

	# sampling for reparameterization 
	def sampling(self, *args):
		mu, log_sigma = args[0]
		epsilon = K.random_normal(shape=(K.shape(mu)))
		return mu + K.exp(log_sigma / 2) * epsilon

class Decoder(AbstractDecoder):
	# symbolic output for decoder
	def build(self, layer_sizes, final_activation=None):
		layer = Concatenate() ([self.latent_input, self.protected_input])
		for l_size in layer_sizes:
			layer = Dense(l_size, activation="relu") (layer)
		layer = Dense(self.feat_dim, activation=final_activation, name="x_hat") (layer)
		return(layer)

class Discriminator(AbstractDiscriminator):
	# symbolic output for discriminator
	def build(self, layer_sizes, final_activation=None):
		layer = Dense(layer_sizes[0]) (self.latent_input)
		for l_size in layer_sizes[1:]:
			layer = Dense(l_size, activation="relu") (layer)
		layer = Dense(self.protected_dim, activation=final_activation, name="x_hat") (layer) # no activation in final layer since regression
		return(layer)






