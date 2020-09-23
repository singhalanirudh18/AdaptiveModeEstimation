import os
import numpy as np
from skimage import io as skio, color as skcolor
tiny_imagenet_test_collection = skio.imread_collection(os.path.expanduser('tiny-imagenet-200/test/images/*.JPEG'))
tiny_imagenet = np.zeros((len(tiny_imagenet_test_collection), 64 * 64 * 3))
for i, img in enumerate(tiny_imagenet_test_collection):
    if img.ndim == 2:
        img = skcolor.gray2rgb(img)
    tiny_imagenet[i, :] = img.ravel()

tiny_imagenet /= tiny_imagenet.max()
tiny_imagenet.shape
np.save('../Tiny_imagenet_dataset.npy',tiny_imagenet)
