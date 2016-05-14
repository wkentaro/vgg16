#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import cPickle as pickle
import glob
import os.path as osp
import re

import numpy as np
import plyvel
from scipy.misc import imread
from scipy.misc import imresize
from skimage.transform import resize
from sklearn.datasets.base import Bunch

import fcn


this_dir = osp.dirname(osp.realpath(__file__))


def transform(img, cropping_size=24, scaling_size=28):
    # cropping
    for offset_y in range(0, 8 + 4, 4):
        for offset_x in range(0, 8 + 4, 4):
            im = img[offset_y:offset_y + cropping_size,
                     offset_x:offset_x + cropping_size]
            # global contrast normalization
            im = im.astype(np.float)
            im -= im.reshape(-1, 3).mean(axis=0)
            im -= im.reshape(-1, 3).std(axis=0) + 1e-5

    # scaling
    for offset_y in range(0, 4 + 2, 2):
        for offset_x in range(0, 4 + 2, 2):
            im = img[offset_y:offset_y + scaling_size,
                     offset_x:offset_x + scaling_size]
            im = imresize(im, (cropping_size, cropping_size),
                          'nearest')
            # global contrast normalization
            im = im.astype(np.float)
            im -= im.reshape(-1, 3).mean(axis=0)
            im -= im.reshape(-1, 3).std(axis=0) + 1e-5


class APC2015(Bunch):

    berkeley_dataset_dir = osp.join(this_dir, 'dataset/APC2015berkeley')

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

        self.ids = []
        self.img_files = []
        self.mask_files = []
        self.target = []

        self._load_berkeley()

        self.ids = np.array(self.ids)
        self.img_files = np.array(self.img_files)
        self.mask_files = np.array(self.mask_files)
        self.target = np.array(self.target)

    def _load_berkeley(self):
        # APC2015berkeley dataset
        for label_value, label_name in enumerate(self.target_names):
            img_file_glob = osp.join(
                self.berkeley_dataset_dir, label_name, '*.jpg')
            for img_file in glob.glob(img_file_glob):
                img_id = re.sub('.jpg$', '', osp.basename(img_file))
                mask_file = osp.join(
                    self.berkeley_dataset_dir, label_name, 'masks',
                    img_id + '_mask.jpg')
                id_ = osp.join('berkeley', label_name, img_id)
                self.ids.append(id_)
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
                self.target.append(label_value)

    @staticmethod
    def _rgb_to_blob(rgb):
        rgb = rgb.astype(np.float64)
        blob = rgb[:, :, ::-1]  # RGB-> BGR
        blob -= np.array((104.00698793, 116.66876762, 122.67891434))
        blob = resize(blob, (224, 224), preserve_range=True)
        blob = blob.transpose((2, 0, 1))
        return blob

    def next_batch(self, batch_size):
        n_data = len(self.ids)
        indices = np.random.randint(0, n_data, batch_size)
        x, t = [], []
        for index in indices:
            id_ = self.ids[index]
            xt = self.db.get(str(id_))
            if xt is None:
                ti = self.target[index]
                img_file = self.img_files[index]
                mask_file = self.mask_files[index]
                img = imread(img_file, mode='RGB')
                mask = imread(mask_file, mode='L')
                img = fcn.util.apply_mask(img, mask, crop=True)
                transform(img)
                xi = self._rgb_to_blob(img)
                xt = {'x': xi, 't': ti}
                self.db.put(str(id_), pickle.dumps(xt))
            else:
                xt = pickle.loads(xt)
            x.append(xt['x'])
            t.append(xt['t'])
        x = np.array(x, dtype=np.float32)
        t = np.array(t, dtype=np.int32)
        return x, t
