#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import tempfile

from chainer import cuda
import chainer.serializers as S
from chainer import Variable
import cv2
import numpy as np
from scipy.misc import imsave
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.color import gray2rgb

import apc2015
import fcn.util
from vgg16 import VGG16


def draw_test_result(dataset, fname, x_data, label_true, label_pred, n_class):
    cmap = fcn.util.labelcolormap(n_class)

    imgs = []
    for i, blob in enumerate(x_data):
        rgb = dataset.blob_to_rgb(blob)
        rgb = rgb.copy()  # FIXME: this is required to put text
        if label_true[i] != label_pred[i]:
            gray = img_as_float(rgb2gray(rgb))
            rgb = np.array([1.0, 0, 0]) * 0.3 + gray2rgb(gray) * 0.7
            rgb = (rgb * 255).astype(np.uint8)
        rgb[:20, :] = cmap[label_pred[i]] * 255
        cv2.putText(rgb, dataset.target_names[label_pred[i]],
                    org=(0, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1)
        imgs.append(rgb)

    result_img = fcn.util.get_tile_image(imgs)
    imsave(fname, result_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    gpu = args.gpu

    save_dir = 'test_result'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    dataset = apc2015.APC2015(tempfile.mktemp())
    n_class = len(dataset.target_names)

    model = VGG16(n_class=n_class)
    S.load_hdf5('snapshot/vgg16_96000.chainermodel', model)
    if gpu != -1:
        model.to_gpu(gpu)

    batch_size = 25
    index = 0
    for index_start in xrange(0, len(dataset.test), batch_size):
        indices = range(index_start,
                        min(len(dataset.test), index_start + batch_size))
        x, t = dataset.next_batch(batch_size, type='test',
                                  type_indices=indices)
        if gpu != -1:
            x = cuda.to_gpu(x, gpu)
            t = cuda.to_gpu(t, gpu)
        x = Variable(x, volatile=True)
        t = Variable(t, volatile=True)
        model(x, t)

        x_data = cuda.to_cpu(x.data)
        accuracy = float(cuda.to_cpu(model.acc.data))
        score = cuda.to_cpu(model.pred.data)
        label_true = cuda.to_gpu(t.data)
        label_pred = score.argmax(axis=1)

        fname = 'test_{0}-{1}_{2:.2}.png'.format(
            indices[0], indices[-1], accuracy)
        fname = osp.join(save_dir, fname)
        draw_test_result(dataset, fname, x_data,
                         label_true, label_pred, n_class)
        print('Saved {0}.'.format(fname))


if __name__ == '__main__':
    main()
