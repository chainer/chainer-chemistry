# " -*- coding: utf-8 -*-"
# ------------------------------------------------------------------------------
# Name:        neighbor_node_expansion
# Purpose:     Implement Neighbor Node Expansion
#
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     04/10/2019 (DD/MM/YY)
# Last update: 04/10/2019 (DD/MM/YY)
# ------------------------------------------------------------------------------

# from io import *
import sys
import os

import numpy as np

import chainer.functions as F

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

DEBUG = False

def apply_nne_for_datasets(datasets):
    """

    :param self:
    :param datasets: tuple of dataset (usually, train/val/test), each dataset consists of atom_array and adj_array and teach_signal
    :return: - tuple of dataset (usually, train/val/test), each dataest consists of atom_number_array and adj_tensor with expanded labels
              - list of all labels, used in the dataset parts.
    """


    #
    # load all atom_array (ndarray int32) and adj_arrays
    #

    trainvaltest_atom_arrays, trainvaltest_adj_arrays, trainvaltest_teach_signals = load_dataset_elements_trainvaltest(datasets)

    #
    # list all expanded labels. at the same time, associate the expanded label with atom nodes.
    #
    trainvaltest_expanded_atom_lists, labels_expanded = list_all_expanded_labels(trainvaltest_adj_arrays,
                                                                               trainvaltest_atom_arrays)
    num_expanded_labels = len(labels_expanded)
    range_num_expanded_labesl = range(num_expanded_labels)
    if DEBUG:
        print("expaneded labels: cardinality=" + str(num_expanded_labels))

    #
    # rexpand the atomic number array, returns the valid datasets (tuple of train/val/test NumpyTupleDataset)
    #
    datasets_expanded = []

    # ToDo: try another indexing: e.g. oirignal node label + extneions
    for set_expanded_atom_lists, set_adj_arrays, set_teach_signals in zip(trainvaltest_expanded_atom_lists,
                                                                          trainvaltest_adj_arrays,
                                                                          trainvaltest_teach_signals):
        dataset_expanded = []  # list of ndarrays.

        # prepare the expanded atom arrays. num_graphs X Nx X 1. one-hot vector = indicator

        # for each graph
        for expanded_atom_list, adj_array, teach_signal in zip(set_expanded_atom_lists, set_adj_arrays,
                                                               set_teach_signals):
            N = len(adj_array)
            assert len(expanded_atom_list) == N
            expanded_atom_array = np.zeros(N)

            # find the index of the expanded label(i). if not founc (-1), it is a error. die
            for i, expanded_label in enumerate(expanded_atom_list):
                idx = labels_expanded.index(expanded_label)

                if DEBUG:
                    print("idx=", idx)
                    print("expanded_label=", expanded_label)

                assert idx in range_num_expanded_labesl  # if assertiona filed, not expanded symbol with the all_expanded_labels
                expanded_atom_array[i] = idx

            # end for

            dataset_expanded.append(expanded_atom_array.astype(np.int32))
        # end set_expanced-for
        expanded_atom_arrays = np.array(dataset_expanded)
        adj_arrays = np.array(set_adj_arrays)
        teach_signals = np.array(set_teach_signals)

        datasets_expanded.append(NumpyTupleDataset(expanded_atom_arrays, adj_arrays, teach_signals))
    # end trainvaltest_expanded-for

    return tuple(datasets_expanded), labels_expanded


def list_all_expanded_labels(trainvaltest_adj_arrays, trainvaltest_atom_arrays):
    trainvaltest_expanded_atom_lists = []  # atom_array values are expanded label "STRING", not numbers
    labels_expanded = []
    # for each train/val/test, do
    for atom_arrays, adj_arrays in zip(trainvaltest_atom_arrays, trainvaltest_adj_arrays):

        # for each molecule sample, do
        set_expanded_atom_list = []
        for atom_array, adj_array in zip(atom_arrays, adj_arrays):

            N = len(atom_array)  # number of molecules
            # atom_array: N by F
            # adj_array: N by N or N by N by R

            # compress the relation axis
            if adj_array.ndim == 3:
                adj_array = np.sum(adj_array, axis=2, keepdims=False)
                assert len(adj_array.shape) == 2
            # end-if
            assert adj_array.shape == (N, N)

            neighbors = np.nonzero(adj_array)  # array[0]: row index array[1]: column index

            mol_expanded_atom_list = []
            # for each node i in the molecule, get the neighbor's atom label (number index)
            for i in range(N):
                expanded_label = str(atom_array[i])
                if DEBUG:
                    print("i=", i, " expanded_label=", expanded_label)

                neighbor_i = neighbors[1][np.where(neighbors[0] == i)]
                if DEBUG:
                    print(neighbor_i)
                    print("len(neighbor_i)=", len(neighbor_i))
                # end-if

                # expanded labels, except the self link
                neighbor_labels = np.sort([atom_array[x] for x in neighbor_i if x != i])
                extension_label = ".".join(map(str, neighbor_labels))
                expanded_label = expanded_label + "-" + extension_label
                if DEBUG:
                    print("neighbor_labels=", neighbor_labels)
                    print("extension_label=" + extension_label)
                    print("expanded_label=" + expanded_label)

                mol_expanded_atom_list.append(expanded_label)
                if expanded_label not in labels_expanded:
                    labels_expanded.append(expanded_label)
            # end i-for
            set_expanded_atom_list.append(mol_expanded_atom_list)
        # end zip(atom_arrays, adj_array)-for

        trainvaltest_expanded_atom_lists.append(set_expanded_atom_list)
    # end zip(trainvaltest_atom_arrays, trainvaltest_adj_array)-for
    return trainvaltest_expanded_atom_lists, labels_expanded


def load_dataset_elements_trainvaltest(datasets):

    if True:
        print('type(datasets)', type(datasets))

    trainvaltest_atom_arrays = []  # 3 by num_mols by N by F
    trainvaltest_adj_arrays = []  # 3 by num_mols by N by N, or 3 by N by N by N R
    trainvaltest_teach_signals = []  # 3 by num_mols by N by (data-dependent)
    for dataset in datasets:

        if DEBUG:
            print('type(dataset)', type(dataset))

        atom_arrays = []  # N by F
        adj_arrays = []  # N by N by N or N by N by N by R
        teach_signals = []  # N by (data-dependent)

        for mol_data in dataset:
            atom_array = mol_data[0]
            adj_array = mol_data[1]
            teach_signal = mol_data[2]

            if DEBUG:
                print("type(atom_arrray)=", type(atom_array))
                print("type(adj_arrray)=", type(adj_array))
                print("type(teach_signal)=", type(teach_signal))

            atom_arrays.append(atom_array)
            adj_arrays.append(adj_array)
            teach_signals.append(teach_signal)
        # end dataset-for

        trainvaltest_atom_arrays.append(atom_arrays)
        trainvaltest_adj_arrays.append(adj_arrays)
        trainvaltest_teach_signals.append(teach_signals)
    # end datasets-for
    return trainvaltest_atom_arrays, trainvaltest_adj_arrays, trainvaltest_teach_signals

# end of apply_nne_for_trainvaltest

