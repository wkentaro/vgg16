#!/usr/bin/env python

import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def learning_curve(csv_file):
    df = pd.read_csv(csv_file)
    df_train  = df.query("type == 'train'")
    df_val = df.query("type == 'test'")

    colors = sns.husl_palette(3, l=.5, s=.5)

    plt.figure(figsize=(12, 6), dpi=500)

    #########
    # TRAIN #
    #########

    # train loss
    plt.subplot(221)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(df_train.i_iter, df_train.loss, '-', markersize=1, color=colors[0],
             alpha=.5, label='train loss')
    plt.xlabel('iteration')
    plt.ylabel('train loss')

    # train accuracy
    plt.subplot(222)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(df_train.i_iter, df_train.acc, '-', markersize=1, color=colors[1],
             alpha=.5, label='train accuracy')
    plt.xlabel('iteration')
    plt.ylabel('train overall accuracy')

    #######
    # VAL #
    #######

    # val loss
    plt.subplot(223)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(df_val.i_iter, df_val.loss, 'o-', color=colors[0],
             alpha=.5, label='val loss')
    plt.xlabel('iteration')
    plt.ylabel('val loss')

    # val accuracy
    plt.subplot(224)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(df_val.i_iter, df_val.acc, 'o-', color=colors[1],
             alpha=.5, label='val accuracy')
    plt.xlabel('iteration')
    plt.ylabel('val overall accuracy')

    plt.savefig('plot.png')
    print('Saved plot.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    args = parser.parse_args()
    learning_curve(args.csv_file)


if __name__ == '__main__':
    main()
