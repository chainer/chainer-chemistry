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
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--out_prefix', default="result_")
    args = parser.parse_args()

    #
    # load the config file in the designated directory
    #

    dataset_name = args.dataset
    task_type = molnet_default_config[dataset_name]['task_type']
    print('task type=\'' + str(task_type) + "\'")

    if task_type=='regression':
        metrics = ['main/MAE', 'main/RMSE']
    elif task_type=='classification':
        metrics = ['test/main/roc_auc']

    x = args.methods

    for metric in metrics:

        y = np.zeros( (len(args.methods), args.runs) )

        for m, method in enumerate(args.methods):
            for run in range(0, args.runs):
                #for run in range(1, args.runs+1):
                with open(os.path.join(args.prefix + "_" + method + "_" + str(run), 'eval_result.json')) as f:
                    result = json.load(f)
                    y[m, run-1,] = result[metric]
                # end with
            # end run-for

        # end method-for

        metric_lastslash = metric.rindex("/")
        metric_name = metric[metric_lastslash+1:]

        # draw figure
        save_evaluation_plot(x, np.mean(y, axis=1), metric, dataset_name, args.out_prefix + metric_name + '.png')
        save_evaluation_plot(x, np.mean(y, axis=1), metric, dataset_name, args.out_prefix + metric_name + '.pdf')

        # output as text. mean/std
        y_mean = np.mean(y, axis=1)
        y_std = np.std(y, axis=1)

        with open(args.out_prefix + "_summary_" + metric_name + ".tsv", "w") as fout:
            for m, method in enumerate(args.methods):
                fout.write(method + "\t" + str(y_mean[m]) + "\t" + str(y_std[m]) + "\n")
            # end-for
        # end with

    # end metric-for

if __name__ == "__main__":
    main()
