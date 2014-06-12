import pylab as pl

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn import preprocessing

import numpy as np


class Layer(object):

    def __init__(self, hierarchy, depth, patch_size, num_features, num_patches, multiplier):
        """
         * depth - hierarchy level (1, 2, 3, etc.)
         * patch_size - number of pixels representing side of the square patch.
           like, 8 (8x8 patches)
         * num_features - how many components to learn
         * multiplier - num of subpatches we break patch into
           (0 for the first level). if 3, patch will contant 3x3 subpatches.
        """
        self.hierarchy = hierarchy
        self.depth = depth
        self.basement_size = patch_size
        self.num_features = num_features
        self.num_patches = num_patches
        self.multiplier = multiplier
        self.learning = MiniBatchDictionaryLearning(
            n_components=num_features, n_iter=3000, transform_algorithm='lasso_lars', transform_alpha=0.5, n_jobs=2)
        self.ready = False

    def get_data(self, data, max_patches=None):
        """
        Extracts raw data from patches.
        """
        max_patches = max_patches or self.num_patches
        if isinstance(data, np.ndarray):
            # one image
            patches = extract_patches_2d(
                data, (self.basement_size, self.basement_size), max_patches=max_patches)
        else:
            patches = []
            # multiple images
            for i in xrange(max_patches):
                idx = np.random.randint(len(data))  # selecting random image
                dx = dy = self.basement_size
                if data[idx].shape[0] <= dx or data[idx].shape[1] <= dy:
                    continue
                x = np.random.randint(data[idx].shape[0] - dx)
                y = np.random.randint(data[idx].shape[1] - dy)
                patch = data[idx][x: x + dx, y: y + dy]
                patches.append(patch.reshape(-1))
            patches = np.vstack(patches)
            patches = patches.reshape(patches.shape[0], self.basement_size, self.basement_size)
        print 'patches', patches.shape
        patches = preprocessing.scale(patches)
        return patches

    def learn(self, data):
        data = data.reshape(data.shape[0], -1)
        self.learning.fit(data)
        self.ready = True

    @property
    def output_size(self):
        return int(np.sqrt(self.num_features))

    @property
    def input_size(self):
        if self.depth == 0:
            return self.basement_size
        else:
            prev_layer = self.hierarchy.layers[self.depth - 1]
            r = prev_layer.output_size * self.multiplier
            return r
        return self._input_size

    @property
    def features(self):
        return self.learning.components_

    # def get_features(self):
    #     # going from up to down
    #     result = []
    #     layers = self.hierarchy.layers[: self.depth][::-1]
    #     if self.depth == 0:
    #         return self.features

    #     previous_layer = self.hierarchy.layers[self.depth - 1]
    #     for feature in self.features:
    #         multiplier = self.multiplier
    #         feature = feature.reshape(self.multiplier * previous_layer.output_size,
    #                                   self.multiplier * previous_layer.output_size,)
    #         for other_layer in layers:
    #             expressed_feature = np.empty((multiplier * other_layer.input_size,
    #                                           multiplier * other_layer.input_size))
    #             enc_n = other_layer.output_size
    #             n = other_layer.input_size
    #             for dx in range(multiplier):
    #                 for dy in range(multiplier):
    #                     encoded_subfeature = feature[dx * enc_n: (dx + 1) * enc_n,
    #                                                  dy * enc_n: (dy + 1) * enc_n]
    #                     prev_patch = np.dot(encoded_subfeature.reshape(-1), other_layer.features)
    #                     expressed_feature[dx * n: (dx + 1) * n, dy * n: (dy + 1) * n] = prev_patch.reshape(n, n)
    #             feature = expressed_feature
    #             multiplier *= other_layer.multiplier
    #         result.append(expressed_feature.reshape(-1))
    #     result = np.vstack(result)
    #     return result

    def get_features(self):
        # going from down to up. these two methods are look like the same
        if self.depth == 0:
            return self.features
        layers = self.hierarchy.layers[1: self.depth + 1]  # down --> up
        features = self.hierarchy.layers[0].features  # to express upper feature

        for i, layer in enumerate(layers, start=1):
            previous_layer = self.hierarchy.layers[i - 1]
            expressed_features = []
            for feature in layer.features:
                n = previous_layer.output_size
                m = int(np.sqrt(features.shape[1]))
                feature = feature.reshape((layer.input_size, layer.input_size))
                expressed_feature = np.empty((layer.multiplier * m,
                                              layer.multiplier * m))
                for dx in range(layer.multiplier):
                    for dy in range(layer.multiplier):
                        subfeature = feature[dx * n: (dx + 1) * n, dy * n: (dy + 1) * n]
                        # now that's previous_layer's code. replace it with reconstruction
                        expressed_subfeature = np.dot(subfeature.reshape(-1), features)
                        expressed_feature[dx * m: (dx + 1) * m, dy * m: (dy + 1) * m] = expressed_subfeature.reshape((m, m))
                expressed_features.append(expressed_feature.reshape(-1))
            features = np.vstack(expressed_features)
        return features


class Hierarchy(object):

    def __init__(self, data):
        self.data = data
        self.layers = []

    def add_layer(self, patch_size, num_features, num_patches, multiplier):
        layer = Layer(self, len(self.layers), patch_size,
                      num_features, num_patches, multiplier)
        self.layers.append(layer)

    def express_data_in_neighbor_layer(self, layer, lower_layer, data, depth):
        # neighbor_layer.basement_size
        n = lower_layer.input_size
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
        code = lower_layer.learning.transform(smallpatches)

        n = lower_layer.output_size
        smallpatches_in_bigpatch = (multiplier) * (multiplier)
        result = []
        counter = 0
        result = np.zeros(
            (code.shape[0] / smallpatches_in_bigpatch, n * multiplier, n * multiplier))
        for i in range(0, code.shape[0], smallpatches_in_bigpatch):
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
                # (1, 0), (2, 1) and so on
                if not lower_layer.ready:
                    raise Exception("Layer {} not ready".format(i))
                result = self.express_data_in_neighbor_layer(
                    upper_layer, lower_layer, result, layer.depth - i)
            return result

    def learn(self):
        for layer in self.layers:
            data = layer.get_data(self.data)
            data = self.express_data(layer, data)
            layer.learn(data)

    def visualize_layer(self, layer_idx, rows=10, cols=10):
        layer = self.layers[layer_idx]
        features = layer.get_features()

        for i, comp in enumerate(features):
            pl.subplot(rows, cols, i + 1)
            pl.imshow(comp.reshape(layer.basement_size, layer.basement_size),
                      cmap=pl.cm.gray_r, interpolation='nearest')
            pl.xticks(())
            pl.yticks(())
        pl.show()
