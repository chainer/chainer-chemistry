# " -*- coding: utf-8 -*-"
# ------------------------------------------------------------------------------
# Name:        test_neighbor_node_expansion
# Purpose:     
#
#              inputs: 
#
#              outputs: 
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     10/10/2019 (DD/MM/YY)
# Last update: 10/10/2019 (DD/MM/YY)
# ------------------------------------------------------------------------------

# from io import *
import sys
import os
from unittest import TestCase

# import time
# import csv
# import glob
# import collections
# from collections import Counter
import pickle

import numpy as np

from chainer_chemistry.dataset.preprocessors import preprocess_method_dict, neighbor_node_expansion
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

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

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

if __name__ == '__main__':
    #test = TestNeighborNodeExpansion()


    def test_main():
        print("start testing!")

        # ToDo: wrie a more formal test...

        #
        # dummy dataset. a single NumpyTupleDatasets
        #

        # contains two mols.
        N_1 = 3
        N_2 = 5

        # one-hot atom labels: 1 tp N
        atom_array_1 = np.arange(N_1)
        atom_array_2 = np.arange(N_2)

        # adj-array, manually
        adj_array_1 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]).astype(np.int) # all connectes. expanded labels is a permutaion of 0,1,2
        # node 0 --> 0-1.2
        # node 1 --> 1-0.2
        # node 2 --> 2-0.1

        adj_array_2 = np.array([[1, 1, 0, 0, 1],
                                [1, 1, 0, 0, 1],
                                [0, 0, 1, 1, 0],
                                [0, 0, 1, 1, 0],
                                [1, 1, 0, 0, 1]]).astype(np.float32)
        # node 0 --> 0-1.4
        # node 1 --> 1-0.4
        # node 2 --> 2-3
        # node 3 --> 3-2
        # node 4 --> 4-0.1

        # supervised labels, dummy
        teach_signal_1 = np.array(1).astype(np.int)
        teach_signal_2 = np.array(0).astype(np.int)

        # concat in a one numpy array!
        atom_arrays = np.array([atom_array_1, atom_array_2])
        adj_arrays = np.array([adj_array_1, adj_array_2])
        teach_signals = np.array([teach_signal_1, teach_signal_2])

        datasets = [NumpyTupleDataset(atom_arrays, adj_arrays, teach_signals), NumpyTupleDataset(atom_arrays, adj_arrays, teach_signals)]

        expanded_datasets, expanded_labels = neighbor_node_expansion.apply_nne_for_datasets(datasets)

        #
        # check the result
        #

        # expanded labels should be: ['0-1.2', '1-0.2', '2-0.1', '0-1.4', '1-0.4', '2-3', '3-2', '4-0.1']
        print(expanded_labels)

        # exapnded node atoms must enumerate all labels
        for dataset in expanded_datasets:
            for mol in dataset:
                print(mol[0])

    test_main()


    self.fail()
