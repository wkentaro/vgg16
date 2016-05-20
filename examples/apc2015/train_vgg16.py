#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp

from chainer import cuda
import chainer.optimizers as O
import chainer.serializers as S
from chainer import Variable
import tqdm

import apc2015
import fcn
import fcn.models
import fcn.util
from vgg16 import VGG16


class Trainer(object):

    def __init__(self, gpu, model, optimizer, dataset, save_dir=None,
                 test_interval=1000, snapshot_interval=4000,
                 max_iteration=100000, batch_size=20):
        self.gpu = gpu
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset

        self.save_dir = 'snapshot' if save_dir is None else save_dir
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.test_interval = test_interval
        self.snapshot_interval = snapshot_interval
        self.max_iteration = max_iteration
        self.batch_size = batch_size

        self.log_file = open(osp.join(self.save_dir, 'log.csv'), 'a')
        print('i_iter,type,loss,acc', file=self.log_file)
        self.csv_templ = '{i_iter},{type},{loss},{acc}'
        self.display_templ = '{i_iter}: type={type}, loss={loss}, acc={acc}'

    def __del__(self):
        self.log_file.close()

    def test(self):
        n_data = len(self.dataset.test)
        sum_loss, sum_acc = 0, 0
        desc = '{0}: test iteration'.format(self.i_iter)
        pbar = tqdm.tqdm(total=n_data, ncols=80, desc=desc)
        for index_start in xrange(0, n_data, self.batch_size):
            pbar.update(self.batch_size)
            index_stop = min(index_start + self.batch_size, n_data)
            type_indices = range(index_start, index_stop)
            loss, acc, actual_batch_size = \
                self.iterate_once(type='test', type_indices=type_indices)
            sum_loss += (loss * actual_batch_size)
            sum_acc += (acc * actual_batch_size)
        pbar.close()
        mean_loss = sum_loss / n_data
        mean_acc = sum_acc / n_data
        return mean_loss, mean_acc

    def iterate_once(self, type, type_indices=None):
        assert type in ('train', 'test')
        x, t = self.dataset.next_batch(batch_size=self.batch_size, type=type,
                                       type_indices=type_indices)
        if self.gpu != -1:
            x = cuda.to_gpu(x, self.gpu)
        x = Variable(x, volatile=not self.model.train)
        if self.gpu != -1:
            t = cuda.to_gpu(t, self.gpu)
        t = Variable(t, volatile=not self.model.train)
        if self.model.train:
            self.optimizer.zero_grads()
            self.optimizer.update(self.model, x, t)
        else:
            self.model(x, t)
        loss = float(cuda.to_cpu(self.model.loss.data))
        acc = float(cuda.to_cpu(self.model.acc.data))
        actual_batch_size = len(x.data)  # sometimes != self.batch_size
        return loss, acc, actual_batch_size

    def snapshot(self):
        S.save_hdf5(osp.join(self.save_dir,
                             'vgg16_{0}.chainermodel'.format(self.i_iter)),
                    self.model)
        S.save_hdf5(osp.join(self.save_dir,
                             'vgg16_optimizer_{0}.h5'.format(self.i_iter)),
                    self.optimizer)

    def run(self, iter_start):
        for i_iter in xrange(iter_start, self.max_iteration, self.batch_size):
            self.i_iter = i_iter

            # snapshot
            if self.i_iter % self.snapshot_interval == 0:
                self.snapshot()

            # test
            if self.i_iter % self.test_interval == 0:
                self.model.train = False
                loss, acc = self.test()
                log = dict(i_iter=self.i_iter, type='test', loss=loss, acc=acc)
                print(self.display_templ.format(**log))
                print(self.csv_templ.format(**log), file=self.log_file)

            # train
            self.model.train = True
            loss, acc, _ = self.iterate_once(type='train')
            log = dict(i_iter=self.i_iter, type='train', loss=loss, acc=acc)
            print(self.display_templ.format(**log))
            print(self.csv_templ.format(**log), file=self.log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--resume', nargs=3,
        metavar=('ITER_START', 'CHAINERMODEL', 'OPTIMIZER_STATE'),
        default=('0', None, None))
    parser.add_argument('--max-iter', type=int, default=100000)
    args = parser.parse_args()

    gpu = args.gpu
    iter_start, chainermodel, optimizer_h5 = args.resume
    iter_start = int(iter_start)
    max_iter = args.max_iter

    this_dir = osp.dirname(osp.realpath(__file__))

    # dataset
    db_path = osp.join(this_dir, 'leveldb')
    dataset = apc2015.APC2015(db_path=db_path)
    n_class = len(dataset.target_names)

    # model
    model = VGG16(n_class=n_class)
    if chainermodel is None:
        # copy weights from pretrained model
        model_path = fcn.setup.download_vgg16_chainermodel()
        vgg16_orig = fcn.models.VGG16()
        S.load_hdf5(model_path, vgg16_orig)
        fcn.util.copy_chainermodel(vgg16_orig, model)
    else:
        S.load_hdf5(chainermodel, model)
    if gpu != -1:
        model.to_gpu(gpu)

    # optimizer
    optimizer = O.Adam()
    optimizer.setup(model)
    if optimizer_h5 is not None:
        S.load_hdf5(optimizer_h5, optimizer)

    trainer = Trainer(
        gpu=gpu,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        max_iteration=max_iter,
    )
    trainer.run(iter_start)


if __name__ == '__main__':
    main()
