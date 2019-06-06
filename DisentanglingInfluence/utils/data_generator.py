"""
A class for building a generator to feed data to a model.
Author: Charlie Marx
"""
import numpy as np


class DataGenerator:
	def __init__(self, inputs, batch_size, mode=tuple):
		""" 
		inputs : A list containing the iterable data factors to generate
		batch_size : The number of instances to produce each time the generator is called
		"""
		self.inputs = inputs
		self.batch_size = batch_size
		self.mode = mode

	def set_mode(self, mode):
		""" Determines the output format of the generator """
		assert mode in [tuple, list, np.array], "Unknown output mode requested!"
		self.mode = mode

	def __next__(self):
		n_instances = len(self.inputs[0])
		assert all([(len(data) == n_instances) for data in self.inputs]), "inputs not all of same length!"

		i = 0
		while True:
			# if wrap-around needed
			if (i + self.batch_size) >= n_instances:
				return self.mode([np.append(data[i:], data[:(i+self.batch_size % n_instances)]) for data in self.inputs])

			# if wrap-around not needed
			else:
				return self.mode([data[i:(i + self.batch_size)] for data in self.inputs])

			i = (i + self.batch_size) % n_instances


def test_data_iterator():
	x = [1,2,3,4,5,6]
	y = [6,5,4,3,2,1]
	z = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
	gen = DataGenerator(inputs=[x,y,z], batch_size=2)
	target = ([1,2],[6,5], [[1,2],[3,4]])

	print("Data iterator returns correct tuple? -- ", next(gen) == target)

	for _ in range(5):
		next(gen)

	print("Data iterator cycles correctly? --", next(gen) == target)

	print(next(gen))
if __name__ == "__main__":
	test_data_iterator()

