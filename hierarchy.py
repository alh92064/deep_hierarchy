import pylab as pl

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import numpy as np


class Layer(object):

    def __init__(self, depth, patch_size, num_features, multiplier):
        """
         * depth - hierarchy level (1, 2, 3, etc.)
         * patch_size - number of pixels representing side of the square patch.
           like, 8 (8x8 patches)
         * num_features - how many components to learn
         * multiplier - num of subpatches we break patch into
           (0 for the first level). if 3, patch will contant 3x3 subpatches.
        """
        # self.hierarchy = hierarchy
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = num_features
        self.multiplier = multiplier
        self.learning = MiniBatchDictionaryLearning(
            n_components=num_features, alpha=1, n_iter=1000)
        self.learning.set_params(
            transform_algorithm='omp', transform_n_nonzero_coefs=2)
        self.ready = False

    def get_data(self, img, max_patches=1000):
        """
        Extracts raw data from patches.
        """
        patches = extract_patches_2d(
            img, (self.patch_size, self.patch_size), max_patches=max_patches)
        patches -= np.mean(patches, axis=0)
        patches /= np.std(patches, axis=0)
        return patches

    def learn(self, data):
        data = data.reshape(data.shape[0], -1)
        self.learning.fit(data)
        self.ready = True

    @property
    def features(self):
        return self.learning.components_

    def infer(self, data, multiplier=None):
        """
        Given a dataset, breaks it into input patches, transforms into
        encoded patches and reshapes into similar dataset again.
        """
        if not multiplier:
            # data is m x n
            # and we're supposing n x 1 is some square patch reshaped
            square_side = int(np.sqrt(data.shape[1]))
            # and that square have to contain an integer number of input
            # patches
            multiplier = square_side / self.patch_size
            assert multiplier * \
                self.patch_size == square_side, "Cannot extract integer square number of patches"


class Hierarchy(object):

    def __init__(self, img):
        self.img = img
        self.layers = []

    def add_layer(self, patch_size, num_features, multiplier):
        layer = Layer(len(self.layers), patch_size, num_features, multiplier)
        self.layers.append(layer)

    def get_layer(self, idx):
        return self.layers[idx]

    def express_data_in_neighbor_layer(self, layer, neighbor_layer, data, depth):
        # neighbor_layer.patch_size
        n = int(np.sqrt(neighbor_layer.features.shape[1]))
        multiplier = layer.multiplier ** depth
        smallpatches = []
        for patch in data:
            for dx in range(multiplier):
                for dy in range(multiplier):
                    smallpatch = patch[dx * n:
                                       (dx + 1) * n, dy * n: (dy + 1) * n]
                    # reshaping in line before passing to decoder, don't forget
                    # to reshape back
                    smallpatch = smallpatch.reshape(-1)
                    smallpatches.append(smallpatch)
        smallpatches = np.vstack(smallpatches)
        code = neighbor_layer.learning.transform(smallpatches)

        # square side of an encoded patch (10x10), i.e., 10
        n = int(np.sqrt(code.shape[1]))
        smallpatches_in_bigpatch = (multiplier) * (multiplier)
        result = []
        counter = 0
        result = np.zeros(
            (code.shape[0] / smallpatches_in_bigpatch, n * multiplier, n * multiplier))
        for i in range(0, code.shape[0], smallpatches_in_bigpatch):
            # result.append(bigpatch.reshape(-1))
            # we've got n small patches in a row
            smallpatches = code[i: i + smallpatches_in_bigpatch]
            # big patch is just multiplier*multiplier small patches
            bigpatch = np.empty((n * multiplier, n * multiplier))
            # maitain quadratic orientation!
            # now, first dim is multiplier x multiplier
            for dx in range(multiplier):
                for dy in range(multiplier):
                    patch_num = multiplier * dx + dy
                    smallpatch = smallpatches[patch_num, :].reshape((n, n))
                    bigpatch[dx * n: (dx + 1) * n, dy * n:
                             (dy + 1) * n] = smallpatch
            result[counter, :, :] = bigpatch
            counter += 1
        return result

    def topo_group(self, patches, layer):
        # reshapes patches into quadratic groups based on layer's multiplier
        counter = 0
        result = []
        n = patches.shape[1]  # or [2]
        while counter < patches.shape[0]:
            new_patch = np.zeros((n * layer.multiplier, n * layer.multiplier))
            for dx in range(layer.multiplier):
                for dy in range(layer.multiplier):
                    new_patch[dx * n: (dx + 1) * n, dy * n:
                              (dy + 1) * n] = patches[counter]
                    result.append(new_patch.reshape(-1))
                    counter += 1
        result = np.vstack(result)
        result = result.reshape(
            result.shape[0], n * layer.multiplier, n * layer.multiplier)
        return result

    def express_data(self, layer, data):
        """
        Returns data ready to be learned by layer.
        For layer 0 that's just layer.get_data(), for else
        we need hierarchical expression.
        """
        if layer.depth == 0:
            return data
        else:
            result = data
            # climp up from layer first
            layers = self.layers[:layer.depth + 1]
            for i, (upper_layer, lower_layer) in enumerate(zip(layers[1:], layers[:-1])):
                if not lower_layer.ready:
                    raise Exception("Layer {} not ready".format(i))
                result = self.express_data_in_neighbor_layer(
                    upper_layer, lower_layer, result, layer.depth - i)
                # if i != layer.depth - 1:
                #     result = self.topo_group(result, upper_layer)
            return result

    def learn(self):
        for layer in self.layers:
            data = layer.get_data(self.img)
            data = self.express_data(layer, data)
            layer.learn(data)

    def visualize_layer(self, layer_idx, rows=10, cols=10):
        layer = self.get_layer(layer_idx)
        if layer.depth == 0:
            features = layer.features
        else:
            # layers = self.layers[:layer.depth + 1]
            # for upper_layer, lower_layer in zip(layers[1:], layers[:-1]):
            #     data = self.express_components(upper_layer, lower_layer)

            # don't forget we're learning features
            layers = self.layers[:layer.depth][::-1]
            # going from top to bottom
            features = layer.features
            multiplier = layer.multiplier
            depth = len(layers) + 1 - layer.depth
            for lower_layer in layers:
                features = self.express_components(features, multiplier, depth, lower_layer)
                print 'expressed', features.shape
                multiplier = lower_layer.multiplier
                depth = len(layers) + 1 - lower_layer.depth

        for i, comp in enumerate(features):
            pl.subplot(rows, cols, i + 1)
            pl.imshow(comp.reshape(layer.patch_size, layer.patch_size),
                      cmap=pl.cm.gray_r, interpolation='nearest')
            pl.xticks(())
            pl.yticks(())
        pl.show()

    # def express_components2(self, upper_layer, lower_layer):
    #     feature_size = int(np.sqrt(upper_layer.features.shape[1]))
    #     for feature in upper_features:
    # don't forget: `feature` is what goes as input/patch to layer
    # f.e., for layer 1 it's 30x30
    # we need to reshape it first
    #         square_feature = feature.reshape(feature_size, feature_size)
    # then: if we're in that method at all, feature is a composite
    #         for dx in range(upper_layer.multiplier):
    #             for dy in range(upper_layer.multiplier):
    #                 smallpatch = comp[dx * n: (dx + 1) * n, dy * n: (dy + 1) * n].reshape(-1)

    def express_components(self, upper_features, upper_multiplier, depth, lower_layer):
        # takes big patch, splits it into small patches, decodes them
        # and combines topographicaly
        multiplier = upper_multiplier ** depth
        print 'multi', multiplier
        expressed = []
        comp_size = int(np.sqrt(upper_features.shape[1]))
        lower_comp_size = int(np.sqrt(lower_layer.features.shape[1]))
        result_size = lower_comp_size * multiplier
        n = int(np.sqrt(upper_features.shape[0]))
        print 'n', n
        m = lower_comp_size
        print 'm', m
        for comp in upper_features:
            # remember the order of our permutations
            # feature was first topographicaly combined, then reshaped
            # now doing exactly the opposite
            # print 'comp', comp.shape
            expressed_comp = np.zeros((result_size, result_size))
            comp = comp.reshape(comp_size, comp_size)  # 30x30
            for dx in range(multiplier):
                for dy in range(multiplier):
                    smallpatch = comp[
                        dx * n: (dx + 1) * n, dy * n: (dy + 1) * n].reshape(-1)
                    # now we're going from code to patch here
                    # i.e., 10x10 --> 30x30
                    # only don't forget it's reshaped flat (100,) --> (900,)
                    # print 'lower features', lower_layer.features.shape

                    smallpatch = np.dot(smallpatch, lower_layer.features)
                    expressed_comp[dx * m: (dx + 1) * m, dy * m: (dy + 1) * m] = smallpatch.reshape(m, m)
                    # smallpatches.append(smallpatch)

            # counter = 0
            # for dx in range(upper_layer.multiplier):
            #     for dy in range(upper_layer.multiplier):
            #         expressed_comp[dx * bottom_size: (dx + 1) * bottom_size, dy * bottom_size: (dy + 1)
            #                        * bottom_size] = smallpatches[counter].reshape(bottom_size, bottom_size)
            #         counter += 1

            # and now expressed_comp is big patch
            expressed.append(expressed_comp.reshape(-1))
        expressed = np.vstack(expressed)
        return expressed
