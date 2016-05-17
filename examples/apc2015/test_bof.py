#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import gzip

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import face
from scipy.misc import imread
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

import fcn.util
import apc2015

# ROS releated
from imagesift import get_sift_keypoints
from imagesift import draw_sift_frames


dataset = apc2015.APC2015('leveldb')

with gzip.open('apc2015_bof.pkl.gz', 'rb') as f:
    bof = pickle.load(f)
    if 'n_jobs' not in bof.nn.__dict__:
        bof.nn.n_jobs = 1

with gzip.open('apc2015_lgr.pkl.gz', 'rb') as f:
    lgr = pickle.load(f)

accuracies = []
batch_size = 10
for index_start in xrange(0, len(dataset.test), batch_size):
    index_end = min(len(dataset.test), index_start + batch_size)
    test_indices = dataset.test[range(index_start, index_end)]

    y_true = []
    descs = []
    for index in test_indices:
        gray = imread(dataset.img_files[index], mode='L')
        if dataset.mask_files[index] is not None:
            mask = imread(dataset.mask_files[index], mode='L')
            gray = fcn.util.apply_mask(gray, mask, crop=True)
        frames, desc = get_sift_keypoints(gray)
        if desc.size == 0:
            continue
        y_true.append(dataset.target[index])
        descs.append(desc)

    X = bof.transform(descs)
    normalize(X, copy=False)

    y_proba = lgr.predict_proba(X)
    y_pred = np.argmax(y_proba, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    print('{0}: accuracy={1}'.format(index_start, acc))
    accuracies.extend([acc] * len(test_indices))

print('Mean Accuracy: {0}'.format(np.array(accuracies).mean()))
