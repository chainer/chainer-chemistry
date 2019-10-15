# " -*- coding: utf-8 -*-"
# ------------------------------------------------------------------------------
# Name:        visuzlie_nle_labels
# Purpose:     A quick script to get histogram of NLE-expanded labels
#
#              inputs: pickle file dumped by train_molnet.py. it contains:
#                      (list of expanded labels) and (dictionary of frequency counts (k,v)=(labe, freq))
#
#              outputs: histogram of frequency-sorted labels
#                       freqiemcy-sorted label list (in text)
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     13/11/2019 (DD/MM/YY)
# Last update: 13/11/2019 (DD/MM/YY)
# ------------------------------------------------------------------------------

# from io import *
import sys
import os
import argparse
# import time
# import csv
# import glob
# import collections
# from collections import Counter
import pickle

import numpy as np


# import scipy
# from scipy import sparse

# import numba

# import pandas as pd
# from pandas import Series, DataFrame

# from sklearn.datasets import make_blobs
# from sklearn import preprocessing
# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import auc_score
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn import svm
# from sklearn.externals import joblib
# from sklearn.cross_validation import KFold

# sys.path.append("/home/ishiguro/lib/python/requests/")
# import requests

# import chainer
# from chainer import cuda
# from chainer import functions as F
# from chainer import links as L

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main(input_pickle, out_prefix):
    """

    :param input_pickle:
    :param out_prefix:
    :return:
    """

    # load the dumped pickle
    (labels_list, labels_frequency) = pickle.load(open(input_pickle, 'rb'))

    # extract counts to array. at the same time sanity check: all labels have a count
    counts_array = np.zeros(len(labels_list))
    for i, label in enumerate(labels_list):
        if label not in labels_frequency.keys():
            print("label " + label + " does not included in counts. Die")
            assert 1==2
        else:
            counts_array[i] = labels_frequency[label]
        # end if-else
    # end labels_list-for

    # sort in counts
    sort_index = np.argsort(counts_array)[::-1] # descnding order
    sorted_labels_array = (np.array(labels_list))[sort_index]
    sorted_counts_array = counts_array[sort_index]

    print(sorted_labels_array[:10])
    print(sorted_counts_array[:10])

    # plot top-20
    fig, ax = plt.subplots()
    #plt.hist(sorted_counts_array, bins=len(sorted_labels_array) )
    plt.bar(sorted_labels_array[:20], sorted_counts_array[:20])
    ax.set_xlabel("labels")
    ax.set_ylabel("frequency")
    ax.set_title("Frequency of Neighbor Expanded Labels")
    plt.savefig(out_prefix + "_top20.png")
    plt.savefig(out_prefix + "_top20.pdf")

    # plot all (no label)
    fig, ax = plt.subplots()
    #plt.hist(sorted_counts_array, bins=len(sorted_labels_array) )
    plt.bar(sorted_labels_array, sorted_counts_array)
    ax.get_xaxis().set_ticks([])
    ax.set_xlabel("labels")
    ax.set_ylabel("frequency")
    plt.yscale("log")
    ax.set_title("Frequency of Neighbor Expanded Labels")
    plt.savefig(out_prefix + ".png")
    plt.savefig(out_prefix + ".pdf")


# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='example of argparse')
    parser.add_argument('input',
                        help="input the pickle file")
    parser.add_argument('out_prefix',
                        help="output prefix. Try \"testout\" ")

    args = parser.parse_args()
    # print(args)

    input = args.input
    out_prefix = args.out_prefix

    last_slash = out_prefix.rfind("/")
    if last_slash > -1:
        outdir = out_prefix[0:last_slash]
        if (not os.path.exists(outdir)):
            os.makedirs(outdir)
        # end if


    main(input, out_prefix)