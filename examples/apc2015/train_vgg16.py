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

import apc2015
import fcn
from fcn.models import VGG16


this_dir = osp.dirname(osp.realpath(__file__))


db_path = osp.join(this_dir, 'leveldb')
dataset = apc2015.APC2015(db_path=db_path)
n_class = len(dataset.target_names)


model_path = fcn.setup.download_vgg16_chainermodel()
model = VGG16()
S.load_hdf5(model_path, model)
del model.fc8
model.add_link('fc8', L.Linear(4096, n_class))
model.to_gpu()


optimizer = O.Adam()
optimizer.setup(model)


save_dir = 'snapshot'
if not osp.exists(save_dir):
    os.makedirs(save_dir)


f = open(osp.join(save_dir, 'log.csv'), 'w')
print('i_iter,loss,acc', file=f)
csv_templ = '{i_iter},{loss},{acc}'
display_templ = '{i_iter}: loss={loss}, acc={acc}'


max_iteration = 100000
batch_size = 10
for i_iter in xrange(0, max_iteration, batch_size):
    batch_data = dataset.next_batch(batch_size=batch_size)
    x = np.array([d['x'] for d in batch_data], dtype=np.float32)
    x = cuda.to_gpu(x)
    x = Variable(x, volatile=False)
    t = np.array([d['t'] for d in batch_data], dtype=np.int32)
    t = cuda.to_gpu(t)
    t = Variable(t, volatile=False)
    optimizer.zero_grads()
    model.train = True
    optimizer.update(model, x, t)
    log = dict(
        i_iter=i_iter,
        loss=float(model.loss.data),
        acc=float(model.acc.data),
    )
    print(display_templ.format(**log))
    print(csv_templ.format(**log), file=f)

f.close()
