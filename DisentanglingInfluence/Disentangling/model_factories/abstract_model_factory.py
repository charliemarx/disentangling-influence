from abc import ABC, abstractmethod

class AbstractEncoder(ABC):
	def __init__(self, feat_input, latent_dim):
		self.feat_input = feat_input
		self.latent_dim = latent_dim
		super().__init__()

	@abstractmethod
	def build(self):
		pass

class AbstractDecoder(ABC):
	def __init__(self, latent_input, protected_input, feat_dim):
		self.latent_input = latent_input
		self.protected_input = protected_input
		self.feat_dim = feat_dim
		super().__init__()

	@abstractmethod
	def build(self):
		pass

class AbstractDiscriminator(ABC):
	def __init__(self, latent_input, protected_dim):
		self.latent_input = latent_input
		self.protected_dim = protected_dim
		super().__init__()

	@abstractmethod
	def build(self):
		pass


