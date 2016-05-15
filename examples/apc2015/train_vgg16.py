#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import os.path as osp

from chainer import cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.serializers as S
from chainer import Variable
import numpy as np
import tqdm

import apc2015
import fcn
import fcn.models
import fcn.util
from vgg16 import VGG16


this_dir = osp.dirname(osp.realpath(__file__))


db_path = osp.join(this_dir, 'leveldb')
dataset = apc2015.APC2015(db_path=db_path)
n_class = len(dataset.target_names)


model_path = fcn.setup.download_vgg16_chainermodel()
vgg16_orig = fcn.models.VGG16()
S.load_hdf5(model_path, vgg16_orig)
del vgg16_orig.fc8
vgg16_orig.add_link('fc8', L.Linear(4096, n_class))

model = VGG16()
fcn.util.copy_chainermodel(vgg16_orig, model)
model.to_gpu()


optimizer = O.Adam()
optimizer.setup(model)


save_dir = 'snapshot'
if not osp.exists(save_dir):
    os.makedirs(save_dir)


f = open(osp.join(save_dir, 'log.csv'), 'w')
print('i_iter,type,loss,acc', file=f)
csv_templ = '{i_iter},{type},{loss},{acc}'
display_templ = '{i_iter}: type={type}, loss={loss}, acc={acc}'


test_interval = 1000
snapshot = 4000
max_iteration = 100000
batch_size = 20
for i_iter in xrange(0, max_iteration, batch_size):
    # -------------------------------------------------------------
    # snapshot
    # -------------------------------------------------------------
    if i_iter % snapshot == 0:
        S.save_hdf5(
            osp.join(save_dir, 'vgg16_{0}.chainermodel'.format(i_iter)),
            model)
        S.save_hdf5(
            osp.join(save_dir, 'vgg16_optimizer_{0}.h5'.format(i_iter)),
            optimizer)
    # -------------------------------------------------------------
    # test
    # -------------------------------------------------------------
    if i_iter % test_interval == 0:
        n_data = len(dataset.test)
        sum_loss, sum_acc = 0, 0
        desc = '{0}: test iteration'.format(i_iter)
        pbar = tqdm.tqdm(total=n_data, ncols=80, desc=desc)
        for index_start in xrange(0, n_data, batch_size):
            pbar.update(batch_size)
            index_stop = min(index_start + batch_size, n_data)
            type_indices = range(index_start, index_stop)
            x, t = dataset.next_batch(
                batch_size=batch_size, type='test', type_indices=type_indices)
            x = cuda.to_gpu(x)
            x = Variable(x, volatile=True)
            t = cuda.to_gpu(t)
            t = Variable(t, volatile=True)
            optimizer.zero_grads()
            model.train = False
            model(x, t)
            sum_loss += float(model.loss.data)
            sum_acc += float(model.acc.data)
        pbar.close()
        log = dict(
            i_iter=i_iter,
            type='test',
            loss=sum_loss / n_data,
            acc=sum_acc / n_data,
        )
        print(display_templ.format(**log))
        print(csv_templ.format(**log), file=f)
    # -------------------------------------------------------------
    # train
    # -------------------------------------------------------------
    x, t = dataset.next_batch(batch_size=batch_size, type='train')
    x = cuda.to_gpu(x)
    x = Variable(x, volatile=False)
    t = cuda.to_gpu(t)
    t = Variable(t, volatile=False)
    optimizer.zero_grads()
    model.train = True
    optimizer.update(model, x, t)
    log = dict(
        i_iter=i_iter,
        type='train',
        loss=float(model.loss.data),
        acc=float(model.acc.data),
    )
    print(display_templ.format(**log))
    print(csv_templ.format(**log), file=f)


f.close()
