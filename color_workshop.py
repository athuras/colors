#!/usr/bin/env python

import classifiers.fst as fst
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def main():
    pass

def train(img_dir, MAX_COLORS=32, MAX_ITER=500, sample=None):
    '''Trains the kmeans on the image (png) at img_dir.

    The starting number of colors in the pallette is  MAX_COLORS
    To limit maximum iterations set MAX_ITER=x
    To set a specific sample size (makes everything faster) set sample=X.

    Returns a KMeans object with k=MAX_COLORS
    '''
    img = mpimg.imread(img_dir)
    s = img.shape
    data = img.reshape((s[0] * s[1], s[2]))
    if sample is not None:
        s_max = min([s[0] * s[1], sample])
        sel = np.random.randint(data.shape[0], size=s_max)
        data = data[sel]

    km = fst.KMeans(data, MAX_COLORS)
    km.execute(MAX_ITER)
    return km

def classify_and_show(km, img_dir):
    '''Use a km object to display both the original and altered version of the image at
    img_dir, must be PNG'''
    img = mpimg.imread(img_dir)
    s = img.shape
    data = img.reshape((s[0] * s[1], s[2]))
    labels = km.classify(data)

    plt.imshow(img)
    plt.figure()
    plt.imshow(km.means[labels].reshape(s))


if __name__ == '__main__':
    main()
