from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os.path as osp

import numpy as np
import plyvel
import scipy.ndimage as ndi
import skimage.transform

import fcn


_this_dir = osp.dirname(osp.realpath(__file__))


class APC2016Dataset(object):

    target_names = np.array([
        'no_object',
        'barkely_hide_bones',
        'cherokee_easy_tee_shirt',
        'clorox_utility_brush',
        'cloud_b_plush_bear',
        'command_hooks',
        'cool_shot_glue_sticks',
        'crayola_24_ct',
        'creativity_chenille_stems',
        'dasani_water_bottle',
        'dove_beauty_bar',
        'dr_browns_bottle_brush',
        'easter_turtle_sippy_cup',
        'elmers_washable_no_run_school_glue',
        'expo_dry_erase_board_eraser',
        'fiskars_scissors_red',
        'fitness_gear_3lb_dumbbell',
        'folgers_classic_roast_coffee',
        'hanes_tube_socks',
        'i_am_a_bunny_book',
        'jane_eyre_dvd',
        'kleenex_paper_towels',
        'kleenex_tissue_box',
        'kyjen_squeakin_eggs_plush_puppies',
        'laugh_out_loud_joke_book',
        'oral_b_toothbrush_green',
        'oral_b_toothbrush_red',
        'peva_shower_curtain_liner',
        'platinum_pets_dog_bowl',
        'rawlings_baseball',
        'rolodex_jumbo_pencil_cup',
        'safety_first_outlet_plugs',
        'scotch_bubble_mailer',
        'scotch_duct_tape',
        'soft_white_lightbulb',
        'staples_index_cards',
        'ticonderoga_12_pencils',
        'up_glucose_bottle',
        'womens_knit_gloves',
        'woods_extension_cord',
    ])
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self):
        self.dataset_dir = osp.realpath(
            osp.join(_this_dir, 'dataset/APC2016jsk'))
        self.train, self.test = self.scrape()
        self.db = plyvel.DB(osp.join(_this_dir, 'leveldb'),
                            create_if_missing=True)

    def scrape(self):
        # train
        train_txt = osp.join(self.dataset_dir, 'train.txt')
        train_dirs = [osp.join(self.dataset_dir, 'all', dir_.strip())
                      for dir_ in open(train_txt, 'r').readlines()]
        train_data = self._scrape_dirs(train_dirs)
        # test
        test_txt = osp.join(self.dataset_dir, 'test.txt')
        test_dirs = [osp.join(self.dataset_dir, 'all', dir_.strip())
                     for dir_ in open(test_txt, 'r').readlines()]
        test_data = self._scrape_dirs(test_dirs)
        return train_data, test_data

    def _scrape_dirs(self, dirs):
        dataset = []
        for dir_ in dirs:
            img_file = osp.join(dir_, 'image.png')
            mask_file = osp.join(dir_, 'mask.png')
            label_file = osp.join(dir_, 'label.txt')
            label_name = open(label_file, 'r').read().strip()
            dataset.append((img_file, mask_file, label_name))
        return dataset

    def rgb_to_blob(self, rgb):
        rgb = rgb.astype(np.float32)
        blob = rgb[:, :, ::-1]  # RGB-> BGR
        blob -= self.mean_bgr
        blob = blob.transpose((2, 0, 1))
        return blob

    def blob_to_rgb(self, blob):
        bgr = blob.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb

    def transform_img(self, img, mask, train):
        img_trans = img.copy()
        if train:
            # translate
            height, width = img_trans.shape[:2]
            translation = (int(0.1 * np.random.random() * height),
                           int(0.1 * np.random.random() * width))
            tform = skimage.transform.SimilarityTransform(
                translation=translation)
            img_trans = skimage.transform.warp(img_trans, tform, mode='edge',
                                               preserve_range=True)
            img_trans = img_trans.astype(np.uint8)
        # apply mask
        img_trans[mask == 0] = self.mean_bgr[::-1]
        img_trans = fcn.util.apply_mask(
            img_trans, mask, crop=True, fill_black=False)
        # resize
        # img_trans, _ = fcn.util.resize_with_min_hw(img_trans, min_hw=256)
        img_trans = skimage.transform.resize(
            img_trans, (224, 224), preserve_range=True).astype(np.uint8)
        return img_trans

    def load_datum(self, datum, train):
        assert isinstance(datum, tuple)
        assert len(datum) == 3
        datum_id = '_'.join(datum)
        inputs = self.db.get(datum_id)
        if inputs is None:
            img_file, mask_file, label_name = datum
            img = ndi.imread(img_file, mode='RGB')
            mask = ndi.imread(mask_file, mode='L')
            # resize mask image
            if img.shape[:2] != mask.shape[:2]:
                print('WARNING: img and mask must have same shape. '
                      'Resizing mask {} to img {}.'
                      .format(img.shape[:2], mask.shape))
                mask = skimage.transform.resize(
                    mask, img.shape[:2], preserve_range=True).astype(np.uint8)
            inputs = (img, mask, label_name)
            self.db.put(datum_id, pickle.dumps(inputs))
        else:
            inputs = pickle.loads(inputs)
        img, mask, label_name = inputs
        img_trans = self.transform_img(img, mask, train=train)
        blob = self.rgb_to_blob(img_trans)
        label_id = np.where(self.target_names == label_name)[0][0]
        return blob, label_id

    def next_datum(self, train, index=None):
        data = self.train if train else self.test
        if index is None:
            index = np.random.randint(0, len(data))
        return self.load_datum(data[index], train=train)

    def next_batch(self, batch_size, type, type_indices=None):
        assert type in ('train', 'test')
        if type_indices is None:
            type_indices = [None] * batch_size
        x, t = [], []
        for index in type_indices:
            blob, target = self.next_datum(train=type == 'train', index=index)
            x.append(blob)
            t.append(target)
        x = np.array(x, dtype=np.float32)
        t = np.array(t, dtype=np.int32)
        return x, t


if __name__ == '__main__':
    from skimage.color import label2rgb
    import matplotlib.pyplot as plt
    dataset = APC2016Dataset()
    for datum in dataset.train:
        img_file, mask_file, label_name = datum
        img = ndi.imread(img_file, mode='RGB')
        mask = ndi.imread(mask_file, mode='L')
        label = (mask != 0).astype(np.int32)
        label_viz = label2rgb(label, img, bg_label=0)
        plt.imshow(label_viz)
        plt.show()
