#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
import glob
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import plyvel
from scipy.misc import imread
from skimage.transform import resize
from sklearn.cross_validation import train_test_split
from sklearn.datasets.base import Bunch

import fcn


this_dir = osp.dirname(osp.realpath(__file__))


class APC2015(Bunch):

    def __init__(self, db_path):
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
                id_ = osp.join('rbo', img_id)
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
                id_ = osp.join('jsk20150428', img_id)
                self.ids.append(id_)
                dataset_index = len(self.ids) - 1
                self.datasets['jsk20150428'].append(dataset_index)
                self.img_files.append(img_file)
                self.mask_files.append(None)
                self.target.append(label_value)

    def rgb_to_blob(self, rgb):
        rgb = rgb.astype(np.float64)
        blob = rgb[:, :, ::-1]  # RGB-> BGR
        blob -= self.mean_bgr
        blob = resize(blob, (224, 224), preserve_range=True)
        blob = blob.transpose((2, 0, 1))
        return blob

    def blob_to_rgb(self, blob):
        bgr = blob.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb

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
            xt = self.db.get(str(id_))
            if xt is None:
                ti = self.target[index]
                img_file = self.img_files[index]
                mask_file = self.mask_files[index]
                img = imread(img_file, mode='RGB')
                if mask_file is not None:
                    mask = imread(mask_file, mode='L')
                    img = fcn.util.apply_mask(img, mask, crop=True,
                                            fill_black=False)
                xi = self.rgb_to_blob(img)
                xt = {'x': xi, 't': ti}
                self.db.put(str(id_), pickle.dumps(xt))
            else:
                xt = pickle.loads(xt)
            x.append(xt['x'])
            t.append(xt['t'])
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
