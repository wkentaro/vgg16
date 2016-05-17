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

with gzip.open('bof_trained_data/apc2015_bof.pkl.gz', 'rb') as f:
    bof = pickle.load(f)
    if 'n_jobs' not in bof.nn.__dict__:
        bof.nn.n_jobs = 1

with gzip.open('bof_trained_data/apc2015_lgr.pkl.gz', 'rb') as f:
    lgr = pickle.load(f)

y_true = []
y_pred = []
for index in dataset.test:
    gray = imread(dataset.img_files[index], mode='L')
    if dataset.mask_files[index] is not None:
        mask = imread(dataset.mask_files[index], mode='L')
        gray = fcn.util.apply_mask(gray, mask, crop=True)
    frames, desc = get_sift_keypoints(gray)
    if desc.size == 0:
        continue
    # for inserted 'background' label at index 0
    y_true.append(dataset.target[index] - 1)

    X = bof.transform([desc])
    normalize(X, copy=False)

    y_proba = lgr.predict_proba(X)[0]
    assert len(y_proba) == len(dataset.target_names[1:])
    y_pred.append(np.argmax(y_proba))

acc = accuracy_score(y_true, y_pred)
print('Mean Accuracy: {0}'.format(acc))
