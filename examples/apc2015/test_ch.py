#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import gzip

import fcn.util
import numpy as np
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import apc2015


class CHClassifier(object):

    def __init__(self):
        with gzip.open('ch_trained_data/rgb.pkl.gz') as f:
            X_train = pickle.load(f)
            y_train = pickle.load(f)
        self.clf = SVC(kernel='linear', probability=True)
        self.clf.fit(X_train, y_train)

    def _compute_color_histogram(self, rgb):
        # Downsample pixel values:
        rgb = rgb // 64
        # Separate RGB channels:
        r, g, b = rgb.transpose((2, 0, 1))
        pixels = 1 * r + 4 * g + 16 * b
        hist = np.bincount(pixels.ravel(), minlength=64)
        hist = hist.astype(float)
        return np.log1p(hist)

    def extract(self, rgb):
        X = np.array([self._compute_color_histogram(rgb)])
        return X

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


if __name__ == '__main__':
    dataset = apc2015.APC2015('leveldb')

    clf = CHClassifier()

    y_true = []
    y_pred = []
    for index in dataset.test:
        # for inserted 'background' label at index 0
        y_true.append(dataset.target[index] - 1)

        rgb = imread(dataset.img_files[index], mode='RGB')
        if dataset.mask_files[index] is not None:
            mask = imread(dataset.mask_files[index], mode='L')
            rgb = fcn.util.apply_mask(rgb, mask, crop=True)

        X = clf.extract(rgb)
        y_proba = clf.predict_proba(X)

        assert len(y_proba[0]) == len(dataset.target_names[1:])
        y_pred.append(np.argmax(y_proba[0]))

    acc = accuracy_score(y_true, y_pred)
    print('Mean Accuracy: {0}'.format(acc))
