#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
import glob
import os.path as osp
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import plyvel
from scipy.misc import imread
import skimage.morphology
import skimage.transform
from sklearn.cross_validation import train_test_split
from sklearn.datasets.base import Bunch

import fcn


this_dir = osp.dirname(osp.realpath(__file__))


class APC2015(Bunch):

    def __init__(self, db_path):
        self.n_transforms = 6
        self.transform_random_range = 0.1
        self.db = plyvel.DB(db_path, create_if_missing=True)

        self.target_names = [
            'background',
            'champion_copper_plus_spark_plug',
            'cheezit_big_original',
            'crayola_64_ct',
            'dr_browns_bottle_brush',
            'elmers_washable_no_run_school_glue',
            'expo_dry_erase_board_eraser',
            'feline_greenies_dental_treats',
            'first_years_take_and_toss_straw_cup',
            'genuine_joe_plastic_stir_sticks',
            'highland_6539_self_stick_notes',
            'kong_air_dog_squeakair_tennis_ball',
            'kong_duck_dog_toy',
            'kong_sitting_frog_dog_toy',
            'kyjen_squeakin_eggs_plush_puppies',
            'laugh_out_loud_joke_book',
            'mark_twain_huckleberry_finn',
            'mead_index_cards',
            'mommys_helper_outlet_plugs',
            'munchkin_white_hot_duck_bath_toy',
            'oreo_mega_stuf',
            'paper_mate_12_count_mirado_black_warrior',
            'rolodex_jumbo_pencil_cup',
            'safety_works_safety_glasses',
            'sharpie_accent_tank_style_highlighters',
            'stanley_66_052',
        ]
        self.mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

        self.ids = []
        self.img_files = []
        self.mask_files = []
        self.target = []

        self.datasets = defaultdict(list)
        self._load_berkeley()
        self._load_rbo()
        self._load_jsk20150428()
        for name, ids in self.datasets.items():
            print('Loaded {0}: {1}'.format(name, len(ids)))
        assert len(self.ids) == len(set(self.ids))

        self.ids = np.array(self.ids)
        self.img_files = np.array(self.img_files)
        self.mask_files = np.array(self.mask_files)
        self.target = np.array(self.target)

        seed = np.random.RandomState(1234)
        indices = np.arange(len(self.ids))
        self.train, self.test = train_test_split(
            indices, test_size=0.2, random_state=seed)

    def _load_berkeley(self):
        """Load APC2015berkeley dataset"""
        dataset_dir = osp.join(this_dir, 'dataset/APC2015berkeley')
        for label_value, label_name in enumerate(self.target_names):
            img_file_glob = osp.join(
                dataset_dir, label_name, '*.jpg')
            for i, img_file in enumerate(glob.glob(img_file_glob)):
                if i % 15 != 0:
                    continue
                img_id = re.sub('.jpg$', '', osp.basename(img_file))
                mask_file = osp.join(
                    dataset_dir, label_name, 'masks',
                    img_id + '_mask.jpg')
                id_ = osp.join('berkeley', label_name, img_id)
                self.ids.append(id_)
                dataset_index = len(self.ids) - 1
                self.datasets['berkeley'].append(dataset_index)
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
                self.target.append(label_value)

    def _load_rbo(self):
        """Load APC2015rbo dataset"""
        dataset_dir = osp.join(this_dir, 'dataset/APC2015rbo/berlin_selected')
        for label_value, label_name in enumerate(self.target_names):
            mask_file_glob = osp.join(dataset_dir, label_name,
                                      '*_{0}.pbm'.format(label_name))
            for mask_file in glob.glob(mask_file_glob):
                img_id = re.sub('_{0}.pbm'.format(label_name), '',
                                osp.basename(mask_file))
                img_file = osp.join(dataset_dir, label_name, img_id + '.jpg')
                id_ = osp.join('rbo', label_name, img_id)
                self.ids.append(id_)
                dataset_index = len(self.ids) - 1
                self.datasets['rbo'].append(dataset_index)
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
                self.target.append(label_value)

    def _load_jsk20150428(self):
        """Load jsk 20150428 dataset"""
        dataset_dir = osp.join(this_dir, 'dataset/jsk20150428')
        for label_value, label_name in enumerate(self.target_names):
            img_file_glob = osp.join(dataset_dir, label_name, '*.jpg')
            for i, img_file in enumerate(glob.glob(img_file_glob)):
                if i % 15 != 0:
                    continue
                img_id = re.sub('.jpg$', '', osp.basename(img_file))
                id_ = osp.join('jsk20150428', label_name, img_id)
                self.ids.append(id_)
                dataset_index = len(self.ids) - 1
                self.datasets['jsk20150428'].append(dataset_index)
                self.img_files.append(img_file)
                self.mask_files.append(None)
                self.target.append(label_value)

    def rgb_to_blob(self, rgb):
        rgb = rgb.astype(np.float32)
        blob = rgb[:, :, ::-1]  # RGB-> BGR
        blob -= self.mean_bgr
        blob = skimage.transform.resize(blob, (224, 224), preserve_range=True)
        blob = blob.transpose((2, 0, 1))
        return blob

    def blob_to_rgb(self, blob):
        bgr = blob.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb

    def _get_inputs(self, index, type):
        """Get inputs with global index (global means self.ids[index] works)"""
        # prepare inputs:
        #   1. load image and mask
        #   2. transform image
        #   3. apply mask to image and crop
        #   4. convert cropped image to blob
        #   5. save to db
        img = imread(self.img_files[index], mode='RGB')
        img, _ = fcn.util.resize_img_with_max_size(
            img, max_size=1000*2000)
        # apply mask if needed
        mask = None
        if self.mask_files[index] is not None:
            mask = imread(self.mask_files[index], mode='L')
            mask, _ = fcn.util.resize_img_with_max_size(
                mask, max_size=1000*2000)
            # opening operation is required mask image with noise
            mask = skimage.morphology.opening(
                mask, selem=skimage.morphology.square(3))
            if mask.sum() == 0:
                print("Skipping '{file}' ({id}) because of no ROI mask"
                      .format(id=id_, file=self.mask_files[index]),
                      file=sys.stderr)
                return
            where = np.argwhere(mask)
            y_start, x_start = where.min(0)
            y_stop, x_stop = where.max(0) + 1
            height = y_stop - y_start
            width = x_stop - x_start
        height, width = img.shape[:2]
        # prepare transformed images
        if mask is None:
            cropped = img.copy()
        else:
            cropped = fcn.util.apply_mask(
                img, mask, crop=True, fill_black=False)
        # restore image without transformation first
        inputs = [self.rgb_to_blob(cropped)]
        for _ in xrange(self.n_transforms):
            if type == 'test':
                # transformation is not necessary for test images
                break
            translation = (
                int(self.transform_random_range *
                    np.random.random() * height),
                int(self.transform_random_range *
                    np.random.random() * width))
            tform = skimage.transform.SimilarityTransform(
                translation=translation)
            img_trans = skimage.transform.warp(
                img, tform, mode='edge', preserve_range=True)
            if mask is None:
                cropped = img_trans.copy()
            else:
                cropped = fcn.util.apply_mask(
                    img_trans, mask, crop=True, fill_black=False)
            inputs.append(self.rgb_to_blob(cropped))
        return inputs

    def next_batch(self, batch_size, type, type_indices=None):
        assert type in ('train', 'test')
        indices = getattr(self, type)
        n_data = len(indices)
        if type_indices is None:
            type_indices = np.random.randint(0, n_data, batch_size)
        type_selected = indices[type_indices]
        x, t = [], []
        for index in type_selected:
            id_ = self.ids[index]
            inputs = self.db.get(str(id_))
            if inputs is not None:
                # use cached data
                inputs = pickle.loads(inputs)
            else:
                inputs = self._get_inputs(index, type=type)
                if inputs is None:
                    continue
                # save to db
                self.db.put(str(id_), pickle.dumps(inputs))
            if type == 'test':
                blob = inputs[0]
            else:
                blob = inputs[np.random.randint(self.n_transforms+1)]
            x.append(blob)
            t.append(self.target[index])
        x = np.array(x, dtype=np.float32)
        t = np.array(t, dtype=np.int32)
        return x, t


if __name__ == '__main__':
    import tempfile
    dataset = APC2015(tempfile.mktemp())
    print('berkeley data:',
          len([id_ for id_ in dataset.ids if id_.startswith('berkeley/')]))
    print('rbo data:',
          len([id_ for id_ in dataset.ids if id_.startswith('rbo/')]))
    x, _ = dataset.next_batch(batch_size=1, type='train')
    x = x[0]
    rgb = dataset.blob_to_rgb(x)
    plt.imshow(rgb)
    plt.show()
