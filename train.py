import matplotlib.image as mpimg

import pylab as pl

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import numpy as np
import os

V1_PATCH_SIZE = 8
V1_COMPS = 100

MULTIPLIER = 3

V2_PATCH_SIZE = V1_PATCH_SIZE * MULTIPLIER
V2_COMPS = 100

V3_PATCH_SIZE = V2_PATCH_SIZE * MULTIPLIER
V3_COMPS = 100

v1learning = MiniBatchDictionaryLearning(n_components=V1_COMPS, alpha=1, n_iter=1000)
v2learning = MiniBatchDictionaryLearning(n_components=V2_COMPS, alpha=1, n_iter=1000)
v3learning = MiniBatchDictionaryLearning(n_components=V3_COMPS, alpha=1, n_iter=1000)

v1learning.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)
v2learning.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)
v3learning.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=2)



def load_images():
    try:
        return np.load('imgs.npy')
    except:
        imgs = []
        for f in os.listdir('data'):
            num = int(f.split('img')[-1].split('.')[0])
            data = mpimg.imread('data/' + f)
            data = data / 256.
            imgs.append((num, np.float32(data)))

        imgs = [t[1] for t in sorted(imgs, key=lambda x: x[0])]
        np.save('imgs', imgs)
    return np.dstack(imgs)



def main():
    # data is width x height x images number array
    images = load_images()
    img = images[0, :, :]

    from hierarchy import Hierarchy

    H = Hierarchy(img)
    multiplier = 3
    H.add_layer(8, 100, 0)
    H.add_layer(8 * multiplier, 100, multiplier)
    H.add_layer(8 * multiplier * multiplier, 100, multiplier)
    H.learn()
    H.visualize_layer(2)



if __name__ == '__main__':
    main()
