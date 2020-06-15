# " -*- coding: utf-8 -*-"
# ------------------------------------------------------------------------------
# Name:        generate_artificial_graphs
# Purpose:     
#
#              inputs: 
#
#              outputs: 
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     23/01/2020 (DD/MM/YY)
# Last update: 23/01/2020 (DD/MM/YY)
# ------------------------------------------------------------------------------

# from io import *
import sys
import os
import argparse
#import commands
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

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import networkx as nx

def main(num_nodes, num_graphs, num_labels, out_prefix):
    """
    Geenerate random graphs, and dump two types of Trainable Datasets (NumPyTupleDatasets) for chainer-chemistry.

    * erdos-renyi graph
    * all nodes must be connected
    * trunctate edges based on valency

    * attach labels
    - 1. check whether a specific node label is attached to many nodes (more than half)
    - 2. check the exeistence of specific sub-graph

    :param num_nodes:
    :param num_graphs:
    :param num_labels:
    :param out_prefix:
    :return:
    """

    g_list = []

    #
    # generate valid graphs
    #

    # erdos-renyi
    max_binom_prob = np.log(num_nodes) / num_nodes # with this prob. all nodes will be connected with h.p.
    print(max_binom_prob)
    for i in range(num_graphs):
        print("graph ", i)

        # randomize binomial prob.
        pr = max_binom_prob * np.random.uniform(1.0, 2.0)
        print(pr)

        # set psudo-valency
        g = nx.erdos_renyi_graph(num_nodes, pr)
        degs = g.degree
        #print(degs)

        list_degs = np.array([ degs[x] for x in range(num_nodes) ])
        #print(list_degs)
        #print(np.any(list_degs > 3))
        if np.any(list_degs > 4):
            print("bad valency")
            continue

        # Adjacency should have self-link (aligining to molnet preprocessors)
        A = nx.to_numpy_array(g) + np.eye(num_nodes)

        # check connetitivity
        dg = nx.DiGraph(g)
        if not nx.is_weakly_connected(dg):
            print("not connected")
            continue
        else:
            print("connected")
            print(A)
            g_list.append(g)
        # end if-else
    # end i-for

    print("valid graphs generated; ", len(g_list))


    #
    # attach node labels.
    # At the same time we compute the supervised signal 1
    # -- check whether a specific node label is attached to many nodes (more than half)
    #
    print("Attach node labels")
    occupy_pr = 1.5 / num_labels
    #occupy_pr = 0.5
    y_list_1 = []
    for g in g_list:

        # random generation of labels
        labels_g = np.random.randint(0, num_labels, size=num_nodes)
        u_elem, u_cnt = np.unique(labels_g,return_counts=True)
        #print(labels_g)
        #print(u_elem)
        #print(u_cnt)

        for i in range(num_nodes):
            g.nodes[i]["atom"] = labels_g[i]

        if np.max(u_cnt) > occupy_pr * num_nodes:
            y_list_1.append([1, 0])
        else:
            y_list_1.append([0, 1])
    #

    #
    # attach supervied signal 2
    # check whether a specific subgraph is included
    #

    y_list_2 = []

    # subgraph patterns

    # path graph
    subg1 = nx.path_graph(3)
    subg1_atoms = [0, 1, 1,]
    for i in range(len(subg1_atoms)):
        subg1.nodes[i]["atom"] = subg1_atoms[i]
    #print(subg1.nodes.data())
    #print(subg1.edges)

    # triangle
    subg2 = nx.path_graph(3)
    subg2_atoms = [0, 1, 2]
    for i in range(len(subg2_atoms)):
        subg2.nodes[i]["atom"] = subg2_atoms[i]
    subg2.add_edge(1,2)
    #print(subg2.nodes.data())
    #print(subg2.edges)

    # cross
    subg3 = nx.path_graph(3)
    subg3.add_node(3)
    subg3.add_node(4)
    subg3_atoms = [0, 1, 1, 1, 2]
    for i in range(len(subg3_atoms)):
        subg3.nodes[i]["atom"] = subg3_atoms[i]
    subg3.add_edge(1,3)
    subg3.add_edge(1,4)
    #print(subg3.nodes.data())
    #print(subg3.edges)

    # triangle + tail
    subg4 = nx.path_graph(5)
    subg4_atoms = [0, 1, 2, 1, 2]
    for i in range(len(subg4_atoms)):
        subg4.nodes[i]["atom"] = subg4_atoms[i]
    subg4.add_edge(0,2)
    #print(subg4.nodes.data())
    #print(subg4.edges)

    # long path
    subg5 = nx.path_graph(5)
    subg5_atoms = [2, 1, 0, 0, 3]
    for i in range(len(subg5_atoms)):
        subg5.nodes[i]["atom"] = subg5_atoms[i]
    #print(subg4.nodes.data())
    #print(subg4.edges)

    #subgraphs_name = "sub1, subg2"
    #subgraphs = [subg1, subg2]
    subgraphs_name = "sub3, subg4, subg5"
    subgraphs = [subg3, subg4, subg5]

    for g in g_list:
        flags = []
        for sg in subgraphs:
            g1g2 = nx.algorithms.isomorphism.GraphMatcher(g, sg,
                                                      node_match=nx.algorithms.isomorphism.numerical_node_match(['atom'],[-1]))
            flags.append(g1g2.subgraph_is_isomorphic())
        # end sg

        if np.any(flags):
            y_list_2.append([1, 0])
        else:
            y_list_2.append([0, 1])
        #
    # end g

    # check
    #for g,y1,y2 in zip(g_list, y_list_1,y_list_2):
    #    print(y1, y2, g.nodes.data())

    #
    # Dump the datasets, as well its stat (number of labels, for example)
    #

    def generate_TranValTest(g_list, y_list):
        from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
        num_train = int(len(g_list) * 0.8)
        num_val = int(len(g_list) * 0.1)
        num_test = len(g_list) - num_train - num_val
        print("Number of train/val/test: ", num_train, num_val, num_test)

        Xs = []
        As = []
        ys = []
        for i in range(0, num_train):
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g))
            ys.append(y)
        dataset_train = NumpyTupleDataset(np.array(Xs, dtype=np.int32), np.array(As, dtype=np.float32), np.array(ys, dtype=np.int32))

        Xs = []
        As = []
        ys = []
        for i in range(num_train, num_train+num_val):
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g))
            ys.append(y)
        dataset_val = NumpyTupleDataset(np.array(Xs, dtype=np.int32), np.array(As, dtype=np.float32), np.array(ys, dtype=np.int32))

        Xs = []
        As = []
        ys = []
        for i in range(num_train+num_val, len(g_list)):
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g))
            ys.append(y)
        dataset_tset = NumpyTupleDataset(np.array(Xs, dtype=np.int32), np.array(As, dtype=np.float32), np.array(ys, dtype=np.int32))

        # number of labels
        y_array = np.array(y_list)
        sum_y = np.sum(y_array, axis=0)
        print(sum_y)

        return [dataset_train, dataset_val, dataset_tset, sum_y]
    # end def


    # dump the dataset 1
    print("Dump the dataset 1")
    train_d, val_d, test_d, sum_y = generate_TranValTest(g_list, y_list_1)
    dataset1 = [train_d, val_d, test_d]
    try:
        out_name = out_prefix + "_data_1.pkl"
        with open(out_name, 'wb') as fout:
            pickle.dump(dataset1, fout)
        print("wrote to ", out_name)

        out_name = out_prefix + "_data_1.stat"
        with open(out_name, 'w') as fout:
            fout.write("num valid graphs: " + str(len(g_list)) + "\n")
            fout.write("num nodes, labels: " + str(num_nodes) + ", " + str(num_labels) + "\n")
            fout.write("occupy_pr: " + str(occupy_pr) + "\n")
            fout.write("label population: " + str(sum_y[0]) + " " + str(sum_y[1]) + "\n")
    except Exception as e:
        print(e)

    # dump the dataset 2
    print("Dump the dataset 2")
    dataset2 = generate_TranValTest(g_list, y_list_2)
    try:
        out_name = out_prefix + "_data_2.pkl"
        with open(out_name, 'wb') as fout:
            pickle.dump(dataset1, fout)
        print("wrote to ", out_name)

        out_name = out_prefix + "_data_2.stat"
        with open(out_name, 'w') as fout:
            fout.write("num valid graphs: " + str(len(g_list)) + "\n")
            fout.write("num nodes, labels: " + str(num_nodes) + ", " + str(num_labels) + "\n")
            fout.write("subgraphs: " + subgraphs_name + "\n")
            fout.write("label population: " + str(sum_y[0]) + " " + str(sum_y[1]) + "\n")
    except Exception as e:
        print(e)

# end of main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='example of argparse')
    parser.add_argument('--num-nodes', type=int, default=8,
                        help="number of nodes in each graph. ")
    parser.add_argument('--num-graphs', type=int, default=5,
                        help="number of graphs to be sampled. ")
    parser.add_argument('--num-labels', type=int, default=3,
                        help="vocabulray size of node labels")
    parser.add_argument('--out-prefix', type=str, default="./temp",
                        help="output prefix. ")

    args = parser.parse_args()
    # print(args)

    num_nodes = args.num_nodes
    num_graphs = args.num_graphs
    num_labels = args.num_labels

    out_prefix = args.out_prefix

    last_slash = out_prefix.rfind("/")
    if last_slash > -1:
        outdir = out_prefix[0:last_slash]
        if (not os.path.exists(outdir)):
            os.makedirs(outdir)
        # end if

    main(num_nodes, num_graphs, num_labels, out_prefix)