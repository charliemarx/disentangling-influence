from DisentanglingInfluence.experiments.dsprite.dsprite_utils import show_images_grid
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


filename = "reduced_dsprites_32.npz"
zip_package = np.load(filename, mmap_mode="r", encoding="bytes")

print(list(zip_package.keys()))

images = zip_package["imgs"]
latents_classes = zip_package["latents_classes"]
latents_values = zip_package["latents_values"]

sample_idxs = np.random.choice(list(range(images.shape[0])), 25)
sample = images[sample_idxs]

show_images_grid(sample)
plt.savefig("sample_small_imgs.png")

