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


def save_evaluation_plot(x, y_mean, metric, dataset_name, filename):
    plt.figure()

    sns.set()
    ax = sns.barplot(y=x, x=y_mean)

    # If "text" does not work, change the attribute name to "s"
    for n, (label, _y) in enumerate(zip(x, y_mean)):
        ax.annotate(
            s='{:.3f}'.format(abs(_y)),
            xy=(_y, n),
            ha='right',
            va='center',
            xytext=(-5, 0),
            textcoords='offset points',
            color='white')

    plt.title('Performance on ' + dataset_name)
    plt.xlabel(metric)
    plt.savefig(filename)

def main():
    parser = argparse.ArgumentParser("Summarize runs")
    parser.add_argument('--indir_prefix', required=True, help="prefix of input directories. "
                                                              "Each dicretory should be named as: {indir_prefix}_{dataset}_{method}_{run index}. "
                                                              "And each direcotyr should contain {dataset}_{method}_all/eval_result.json. ")
    parser.add_argument('--methods', nargs='+', required=True, help="enumerate methods")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tasktype', type=str, required=True)
    parser.add_argument('--runs', type=int, default=10)
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

    dataset_name = args.dataset
    task_type = args.tasktype
    print('task type=\'' + str(task_type) + "\'")

    if task_type=='regression':
        metrics = ['main/MAE', 'main/MSE']
    elif task_type=='classification':
        metrics = ['test/main/roc_auc']

    x = args.methods

    for metric in metrics:

        y = np.zeros( (len(args.methods), 3) ) # 3 for mean, std. and vaild runs

        for m, method in enumerate(args.methods):

            y_method = [] # list

            for run in range(0, args.runs):

                # check there is a result directory
                result_dir = args.indir_prefix + "_" + str(args.dataset) + "_" + str(method) + "_" + str(run) + "/"
                print("result_dir=" + result_dir)
                #print(result_dir)
                #assert os.path.isdir(result_dir)

                # check the inner directory
                result_dir_inner = result_dir + "/" + str(args.dataset) + "_" + str(method) + "_all/"
                #assert os.path.isdir(result_dir_inner)

                #for run in range(1, args.runs+1):
                fname = os.path.join(result_dir_inner, 'eval_result.json')
                if os.path.exists(fname):
                    with open(fname) as f:
                        result = json.load(f)

                        # check keys
                        keys = result.keys()

                        if metric in keys:
                            y_method.append(result[metric])
                        else:
                            print("key: " + metric + " does not exist in " + str(fname) + ": pass")
                        # end if
                    # end with
                else:
                    print(str(fname) + " does not exists: pass")
            # end run-for
            # store mean, std, and vaild_runs
            if len(y_method) > 0:
                y[m, 0] = np.mean(y_method)
                y[m, 1] = np.std(y_method)
                y[m, 2] = len(y_method)
        # end method-for

        metric_lastslash = metric.rindex("/")
        metric_name = metric[metric_lastslash+1:]

        # draw figure
        save_evaluation_plot(x, y[:, 0], metric, dataset_name, args.out_prefix + metric_name + '.png')
        save_evaluation_plot(x, y[:, 0], metric, dataset_name, args.out_prefix + metric_name + '.pdf')

        # output as text. mean/std
        with open(args.out_prefix + "_summary_" + metric_name + ".tsv", "w") as fout:
            for m, method in enumerate(args.methods):
                fout.write(method + "\t" + str(y[m,0]) + "\t" + str(y[m,1]) + "\t" + str(y[m,2]) + "\n")
            # end-for
        # end with

    # end metric-for

if __name__ == "__main__":
    main()
