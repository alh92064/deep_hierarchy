import matplotlib.image as mpimg
import numpy as np
import argparse
import os

from hierarchy import Hierarchy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_file(name):
    try:
        data = mpimg.imread(name)
    except IOError:
        raise Exception('{} is not an image file'.format(name))
    if len(data.shape) == 3:
        # rgb, converting to grayscale
        data = np.mean(data, -1)
    if np.max(data) > 1:
        # if pixel values are distributed from 0 to 256
        data = data / 256.
    data = np.float32(data)
    return data


def load_folder(name):
    files = os.listdir(name)
    data = []
    for f in files:
        try:
            data.append(load_file(os.path.join(name, f)))
        except Exception as e:
            print e
    if not data:
        raise Exception("{} if empty or contains no image files".format(name))
    return data


def main(input_data, basement, num_layers, multiplier, num_patches, features):
    if os.path.isfile(input_data):
        data = load_file(input_data)
    elif os.path.isdir(input_data):
        data = load_folder(input_data)
    else:
        raise Exception('{} is neither file nor directory'.format(input_data))
    if features:
        try:
            features = [int(f) for f in features.split(',')]
        except:
            raise Exception('Bad features: {}. Expecting a bunch of integers separated by comma'.format(features))
        if len(features) != num_layers:
            raise Exception('{} features specified, but total number of layers is {}'.format(len(features), num_layers))

    H = Hierarchy(data)
    for i in xrange(num_layers):
        mult = 0 if i == 0 else multiplier
        f = features[i] if features else 100
        H.add_layer(basement * (multiplier ** i), f, num_patches, mult)
    H.learn()
    H.visualize_layer(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help='image or folder with images')
    parser.add_argument("-b", "--basement", type=int, help='patch size for the bottom layer', default=8)
    parser.add_argument("-l", "--num_layers", type=int, help='number of layers in hierarchy', default=3)
    parser.add_argument("-m", "--multiplier", type=int, help='layer multiplier', default=3)
    parser.add_argument("--num_patches", type=int, help='number of patches to train on', default=1000)
    parser.add_argument("--features", type=str, help='number of features for each layer. Example: 25,81,100 for 3 layers')
    args = parser.parse_args()
    main(args.input, args.basement, args.num_layers, args.multiplier, args.num_patches, args.features)
