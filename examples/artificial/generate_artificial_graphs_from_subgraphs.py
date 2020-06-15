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

    * 4-regular graph thus assure valency
    * all nodes must be connected
    * randomly erase some of edges

    * attach labels
    - 1. check whether a specific node label is attached to many nodes (more than half)
    - 2. check the exeistence of specific sub-graph

    :param num_nodes:
    :param num_graphs:
    :param num_labels:
    :param out_prefix:
    :return:
    """

    #
    # subgraph patterns
    #

    # 4-circle+1
    subg3 = nx.path_graph(4)
    subg3.add_node(4)
    subg3_atoms = [0, 1, 2, 3, 0]
    for i in range(len(subg3_atoms)):
        subg3.nodes[i]["atom"] = subg3_atoms[i]
    subg3.add_edge(0,3) # 0-1-2-3(-0)
    subg3.add_edge(1,4)
    #print(subg3.nodes.data())
    #print(subg3.edges)

    # triangle+2
    subg4 = nx.path_graph(5)
    subg4_atoms = [0, 1, 2, 3, 0]
    for i in range(len(subg4_atoms)):
        subg4.nodes[i]["atom"] = subg4_atoms[i]
    subg4.add_edge(0,2) # 0-1-2(-0)
    #print(subg4.nodes.data())
    #print(subg4.edges)

    # 5-circle
    subg5 = nx.path_graph(5)
    subg5_atoms = [0, 1, 2, 3, 0]
    for i in range(len(subg5_atoms)):
        subg5.nodes[i]["atom"] = subg5_atoms[i]
    subg5.add_edge(0,4)
    #print(subg4.nodes.data())
    #print(subg4.edges)

    subgraphs_name = "sub3, subg4, subg5"
    subgraphs = [subg3, subg4, subg5]

    remove_pr = 0.25
    add_pr = 0.25

    g_false_list = []
    g_true_list = []

    def check_iso(g, known_graphs):
        iso_flag = False
        for gg in known_graphs:
            g1g2 = nx.algorithms.isomorphism.GraphMatcher(g, gg, node_match=nx.algorithms.isomorphism.numerical_node_match(['atom'],[-1]))

            if g1g2.is_isomorphic():
                iso_flag = True
                print("found ISO")
                break
            # end-if
        # end-for
        return iso_flag
    # end-def

    #
    # generate valid graphs for negative samples
    #

    # erdos-renyi
    max_binom_prob = np.log(num_nodes) / num_nodes # with this prob. all nodes will be connected with h.p.
    print(max_binom_prob)
    for i in range( num_graphs // 2):
        #print("graph ", i)

        # erdos: randomize binomial prob.
        #pr = max_binom_prob * np.random.uniform(1.0, 2.0)
        #g = nx.erdos_renyi_graph(num_nodes, pr)

        # set psudo-valency
        #degs = g.degree
        #list_degs = np.array([ degs[x] for x in range(num_nodes) ])
        #if np.any(list_degs > 4):
        #    #print("bad valency")
        #    continue

        # 4-regular graph
        g = nx.random_regular_graph(4, num_nodes)

        # randomly erase edges
        for i in range(num_nodes):
            neighbors = g.neighbors(i)
            remove_edge_j = []
            for j in neighbors:
                if np.random.rand() < remove_pr:
                    remove_edge_j.append(j)
                # end-if
            # end-for
            for j in remove_edge_j:
                g.remove_edge(i,j)
            # end-for
        # end i-for

        # check connectitivity
        dg = nx.DiGraph(g)
        if not nx.is_weakly_connected(dg):
            #print("not connected")
            continue

        # else
        #print("connected")

        # else
        # attach node labels.
        labels_g = np.random.randint(0, num_labels, size=num_nodes)
        for i in range(num_nodes):
            g.nodes[i]["atom"] = labels_g[i]
        # end i-for

        # sort out graphs of sub-graph included
        flags = []
        for sg in subgraphs:
            g1g2 = nx.algorithms.isomorphism.GraphMatcher(g, sg, node_match=nx.algorithms.isomorphism.numerical_node_match(['atom'],[-1]))
            flags.append(g1g2.subgraph_is_isomorphic())
        # end sg
        if np.any(flags):
            #print("happend to positive")
            continue

        #else:

        # check isomorphism for all graphs so fa
        if not check_iso(g, g_false_list):
            g_false_list.append(g)
    # end g_list for

    # for all pairs, chech isomorphism

    print("valid but possibly false graphs generated; ", len(g_false_list))

    #
    # Generate vaild and correct graphs from subgraph seeds: add 5 nodes randomly, add edges randomly
    #

    for i in range(num_graphs // 2):
        #print("graph ", i)

        # choose one of three graphs
        g1 = subgraphs[ i % 3 ].copy()
        #print(g.nodes.data())

        # generate 4-regular graph for remaining, then erase some of edges
        g2 = nx.random_regular_graph(4, num_nodes-5)
        for i in range(num_nodes-5):
            neighbors = g2.neighbors(i)
            remove_edge_j = []
            for j in neighbors:
                if np.random.rand() < remove_pr:
                    remove_edge_j.append(j)
                # end-if
            # end-for
            for j in remove_edge_j:
                g2.remove_edge(i,j)
            # end-for
        # end i-for

        # relabel to add
        def mapping(x):
            return x+5
        g22 = nx.relabel_nodes(g2,mapping)

        # combine them
        g = nx.compose(g1, g22)

        # add nodes with random labels
        labels_g = np.random.randint(0, num_labels, size=num_nodes)
        for i in range(5,num_nodes):
            g.nodes[i]["atom"] = labels_g[i]
        #print(g.nodes.data())

        # add edges randomly
        for i in range(5, num_nodes):
            for j in range(5):
                if g.degree[i] < 4 and g.degree[j] < 4 and np.random.rand() < add_pr:
                    g.add_edge(j,i)
        # end i-for

        degs = g.degree
        list_degs = np.array([ degs[x] for x in range(num_nodes) ])
        #print(list_degs)
        if np.any(list_degs > 4):
            #print("bad valency")
            #print(np.where(list_degs > 4))
            continue

        # check connetitivity
        dg = nx.DiGraph(g)
        if not nx.is_weakly_connected(dg):
            #print("not connected")
            continue
        #else:
            #print("connected")

        if not check_iso(g, g_true_list):
            g_true_list.append(g)
        # end if-else
    # end i-for

    print("valid and true graphs generated; ", len(g_true_list))

    #
    # Align the number of pos-neg lists then merge
    #

    min_len = min(len(g_false_list), len(g_true_list))
    print("for each pos/neg list, we take ", min_len, " graphs")
    g_list = g_false_list[:min_len]
    g_list.extend(g_true_list[:min_len])

    print("valid gtaphs generated: ", len(g_list))

    #
    # Supervised signal 1: occupied by single atom
    #
    y_list_1 = []
    occupy_pr = 1.5 / num_labels
    #occupy_pr = 0.5
    for g in g_list:

        labels_g = np.array([g.nodes[i]["atom"] for i in range(num_nodes)])
        u_elem, u_cnt = np.unique(labels_g, return_counts=True)
        #print(labels_g)
        #print(u_elem)
        #print(u_cnt)

        if np.max(u_cnt) > occupy_pr * num_nodes:
            y_list_1.append([1])
        else:
            y_list_1.append([0])
    #

    #
    # attach supervied signal 2
    # check whether a specific subgraph is included
    #
    y_list_2 = []
    for g in g_list:
        flags = []
        for sg in subgraphs:
            g1g2 = nx.algorithms.isomorphism.GraphMatcher(g, sg,
                                                      node_match=nx.algorithms.isomorphism.numerical_node_match(['atom'],[-1]))
            if g1g2.subgraph_is_isomorphic():
                flags.append(1)
            else:
                flags.append(0)
        # end sg

        y_list_2.append(flags)
        #print(flags)
        #if np.any(flags):
        #    y_list_2.append([1])
        #else:
        #    y_list_2.append([0])
        #

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

        num_samples_foreach = int(len(g_list) / 2)

        permute_idx = np.random.permutation(range(num_samples_foreach))

        num_train = int(num_samples_foreach * 0.8)
        num_val = int(num_samples_foreach * 0.1)
        num_test = num_samples_foreach - num_train - num_val
        print("Number of train/val/test: ", num_train*2, num_val*2, num_test*2)

        # Adjacency should have self-link (aligining to molnet preprocessors)
        Xs = []
        As = []
        ys = []
        for ii in range(0, num_train):
            # negative
            i = permute_idx[ii]
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g) + np.eye(num_nodes))
            ys.append(y)

            # positive
            i = permute_idx[ii] + num_samples_foreach
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g) + np.eye(num_nodes))
            ys.append(y)
        dataset_train = NumpyTupleDataset(np.array(Xs, dtype=np.int32), np.array(As, dtype=np.float32), np.array(ys, dtype=np.int32))

        Xs = []
        As = []
        ys = []
        for ii in range(num_train, num_train+num_val):
            # negative
            i = permute_idx[ii]
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g) + np.eye(num_nodes))
            ys.append(y)

            # positive
            i = permute_idx[ii] + num_samples_foreach
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g) + np.eye(num_nodes))
            ys.append(y)
        dataset_val = NumpyTupleDataset(np.array(Xs, dtype=np.int32), np.array(As, dtype=np.float32), np.array(ys, dtype=np.int32))

        Xs = []
        As = []
        ys = []
        for ii in range(num_train+num_val, num_samples_foreach):
            # negative
            i = permute_idx[ii]
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g) + np.eye(num_nodes))
            ys.append(y)

            # positive
            i = permute_idx[ii] + num_samples_foreach
            g = g_list[i]
            y = y_list[i]

            X_g = [ g.nodes[i]["atom"] for i in range(num_nodes)  ]
            Xs.append(X_g)
            As.append(nx.to_numpy_array(g) + np.eye(num_nodes))
            ys.append(y)
        dataset_tset = NumpyTupleDataset(np.array(Xs, dtype=np.int32), np.array(As, dtype=np.float32), np.array(ys, dtype=np.int32))

        # number of labels
        y_array = np.array(y_list)
        sum_y = np.sum(y_array, axis=0)
        print(sum_y)

        return [dataset_train, dataset_val, dataset_tset, sum_y]
    # end def


    out_file_prefix = out_prefix \
                      + "G" + str(len(g_list)) \
                      + "N" + str(num_nodes) \
                      + "L" + str(num_labels)

    # dump the dataset 1
    print("Dump the dataset 1")
    train_d, val_d, test_d, sum_y = generate_TranValTest(g_list, y_list_1)
    dataset1 = [train_d, val_d, test_d]
    try:
        out_name = out_file_prefix + "_data_1.pkl"
        with open(out_name, 'wb') as fout:
            pickle.dump(dataset1, fout)
        print("wrote to ", out_name)

        out_name = out_file_prefix + "_data_1.stat"
        with open(out_name, 'w') as fout:
            fout.write("num valid graphs: " + str(len(g_list)) + "\n")
            fout.write("num nodes, labels: " + str(num_nodes) + ", " + str(num_labels) + "\n")
            fout.write("occupy_pr: " + str(occupy_pr) + "\n")
            fout.write("label population: " + str(sum_y[0]) + " " + str(sum_y[1]) + "\n")
    except Exception as e:
        print(e)

    # dump the dataset 2
    print("Dump the dataset 2")
    train_d, val_d, test_d, sum_y = generate_TranValTest(g_list, y_list_2)
    dataset2 = [train_d, val_d, test_d]
    try:
        out_name = out_file_prefix + "_data_2.pkl"
        with open(out_name, 'wb') as fout:
            pickle.dump(dataset2, fout)
        print("wrote to ", out_name)

        out_name = out_file_prefix + "_data_2.stat"
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