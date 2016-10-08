#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fcn.util
import numpy as np
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import yaml

import apc2015
from test_bof import BoFClassifier
from test_ch import CHClassifier


def main():
    dataset = apc2015.APC2015('leveldb')

    weights = yaml.load(open('boosting_trained_data/classifier_weight.yml', 'r'))
    bof_weights = np.array(
        [weights[obj]['bof']
         for obj in dataset.target_names if obj != 'background'])
    ch_weights = np.array(
        [weights[obj]['color']
         for obj in dataset.target_names if obj != 'background'])

    clf1 = BoFClassifier()
    clf2 = CHClassifier()

    y_true = []
    y_pred = []
    for index in dataset.test:
        rgb = imread(dataset.img_files[index], mode='RGB')
        if dataset.mask_files[index] is not None:
            mask = imread(dataset.mask_files[index], mode='L')
            rgb = fcn.util.apply_mask(rgb, mask, crop=True)

        X1 = clf1.extract(rgb)
        if X1 is None:
            continue
        y_proba1 = clf1.predict_proba(X1)

        X2 = clf2.extract(rgb)
        if X2 is None:
            continue
        y_proba2 = clf2.predict_proba(X2)

        # for inserted 'background' label at index 0
        y_true.append(dataset.target[index] - 1)

        y_proba = bof_weights * y_proba1 + ch_weights * y_proba2
        y_pred.append(np.argmax(y_proba[0]))

    acc = accuracy_score(y_true, y_pred)
    print('Mean Accuracy: {0}'.format(acc))


if __name__ == '__main__':
    main()
