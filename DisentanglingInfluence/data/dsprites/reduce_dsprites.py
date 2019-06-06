"""
Reduces the dimensionality of the dsprites dataset to a specified dimension. 
"""

from DisentanglingInfluence.path import get_path

import numpy as np
from PIL import Image

# source for reading massive file: https://stackoverflow.com/questions/42727412/efficient-way-to-partially-read-large-numpy-file/42727761


def reduce_dsprites(new_dimension=16, size_minimum=3):
	path = get_path()

	print('Generating a {0}x{0} version of the dsprites dataset...'.format(new_dimension))
	
	filename = path + "data/dsprites/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
	zip_package = np.load(filename, mmap_mode="r", encoding="bytes")
	
	images = zip_package["imgs"]
	latents_classes = zip_package["latents_classes"]
	latents_values = zip_package["latents_values"]
	
	large_enough = latents_classes[:,2] >= size_minimum
	
	images = images[large_enough]
	latents_classes = latents_classes[large_enough]
	latents_values = latents_values[large_enough]
	
	small_imgs = []
	for img in images:
		pil_image = Image.fromarray(img, mode="L")
		pil_image = pil_image.resize((new_dimension,new_dimension), resample=Image.LANCZOS)
		small_img = np.array(pil_image)
		small_imgs.append(small_img)
	
	np.savez(path + "data/dsprites/reduced_dsprites_{}.npz".format(new_dimension), imgs=np.array(small_imgs), latents_classes=latents_classes, latents_values=latents_values)

