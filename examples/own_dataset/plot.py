#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns


def save_evaluation_plot(x, y, metric, filename):
    plt.figure()

    sns.set()
    ax = sns.barplot(y=x, x=y)

    for n, (label, _y) in enumerate(zip(x, y)):
        ax.annotate(
            '{:.3f}'.format(abs(_y)),
            xy=(_y, n),
            ha='right',
            va='center',
            xytext=(-5, 0),
            textcoords='offset points',
            color='white')

    plt.title('Performance on own dataset')
    plt.xlabel(metric)
    plt.savefig(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    args = parser.parse_args()

    metrics = ['mean_abs_error', 'root_mean_sqr_error']
    x = args.methods
    y = {metric: [] for metric in metrics}

    for method in args.methods:
        with open(os.path.join(args.prefix + method, 'eval_result.json')) as f:
            result = json.load(f)
            for metric in metrics:
                y[metric].append(result['main/' + metric])

    for metric in metrics:
        save_evaluation_plot(
            x, y[metric], metric, 'eval_' + metric + '_own.png')


if __name__ == "__main__":
    main()
