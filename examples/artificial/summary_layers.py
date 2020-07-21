#! -*- coding: utf-8 -*-
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns

import numpy as np


from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from pandas import DataFrame


def save_evaluation_plot(y, methods, layers, metric, dataset_name, filename):
    # y = [len(methods), len(layers), 3 (0=mean, 1=s.t.d., 2=num)]

    y_mean = y[:, :, 0]
    y_std = y[:, :, 1]

    fix, ax = plt.subplots()

    # OK, use matplotlib...
    fmt_strs = ["--", "-"]
    for m_idx in range(len(methods)):
        plt.errorbar( layers, y_mean[m_idx], yerr=y_std[m_idx], fmt=fmt_strs[m_idx])
    ax.legend(methods)

    plt.title('Performance on ' + dataset_name)
    plt.ylabel(metric)
    plt.xlabel("layers")
    plt.savefig(filename)

def main():
    parser = argparse.ArgumentParser("Summarize layer results, for naive embedding and WL embedding")
    parser.add_argument('--indir', required=True, help="the parent ditctory which saves all result directories. "
                                                              "Each result dicretory should be named as: {indir}/eval_layer{layer}_{dataset prefix}_{method}_{run index}. "
                                                              "And each direcotyr should contain {dataset}_{method}_all/eval_result.json. ")
    parser.add_argument('--methods', nargs='+', required=True, type=str, help="enumerate methods (rsgcn rsgcn_cnle)")
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--tasktype', type=str, required=True)
    parser.add_argument('--layers', nargs='+', required=True, help="enumerate layers like 1 2 3 4 5 6", type=int)
    parser.add_argument('--out_prefix', default="result")
    args = parser.parse_args()

    outprefix_lastslash = args.out_prefix.rindex("/")
    if outprefix_lastslash > -1:
        outdir = args.out_prefix[:outprefix_lastslash]

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    #
    # load the config file in the designated directory
    #

    indir = args.indir
    dataset_names = args.datasets
    task_type = args.tasktype
    layers = args.layers
    methods = args.methods

    print('task type=\'' + str(task_type) + "\'")
    print('layers:', layers, " type(layers):", type(layers))
    print('methods:', methods, " type(methods):", type(methods))

    if task_type=='regression':
        metrics = ['main/MAE', 'main/MSE']
    elif task_type=='classification':
        metrics = ['test/main/roc_auc']
    # end if-else

    #
    # Retrieve all scores for each method
    #
    metric_results = []
    for metric in metrics:
        y = np.zeros( (len(methods), len(layers), 3) ) # 3 for mean, std. and vaild runs

        for m, method in enumerate(args.methods):
            for l_idx, layer in enumerate(layers):
                y_existing_runs = []
                for dataset_name in dataset_names:

                    for run in range(0, args.runs):

                        # check there is a result directory
                        result_dir = indir + "/eval_layer" + str(layer) + "_" + str(dataset_name) + "_" + str(method) + "_" + str(run) + "/"
                        print("result_dir=" + result_dir)

                        # check the inner directory
                        result_dir_inner = result_dir + "/" + str(dataset_name) + "_" + str(method) + "_all/"

                        # for run in range(1, args.runs+1):
                        fname = os.path.join(result_dir_inner, 'eval_result.json')
                        if os.path.exists(fname):
                            with open(fname) as f:
                                result = json.load(f)
                            # end-with

                            # check keys
                            keys = result.keys()

                            if metric in keys:
                                y_existing_runs.append(result[metric])
                            else:
                                print("key: " + metric + " does not exist in " + str(fname) + ": pass")
                            # end if
                        else:
                            print(str(fname) + " does not exists: pass")
                        # end exist-ifelse
                    # end run-for
                # end dataset-for

                # store mean, std, and vaild_runs
                if len(y_existing_runs) > 0:
                    y[m, l_idx, 0] = np.mean(y_existing_runs)
                    y[m, l_idx, 1] = np.std(y_existing_runs)
                    y[m, l_idx, 2] = len(y_existing_runs)
                # end-if
            # end layer-for
        # end method-for

        metric_lastslash = metric.rindex("/")
        metric_name = metric[metric_lastslash+1:]

        # draw figure
        save_evaluation_plot(y, methods, layers, metric, dataset_name, args.out_prefix + metric_name + '.png')
        save_evaluation_plot(y, methods, layers, metric, dataset_name, args.out_prefix + metric_name + '.pdf')

        # output as text. mean/std
        with open(args.out_prefix + "_summary_" + metric_name + ".tsv", "w") as fout:
            for m, method in enumerate(methods):
                for l, layer in enumerate(layers):
                    fout.write(method + "\t" + str(layer) + "\t" + str(y[m,l,0]) + "\t" + str(y[m,l,1]) + "\t" + str(y[m,l,2]) + "\n")
            # end-for
        # end with

    # end metric-for

if __name__ == "__main__":
    main()
