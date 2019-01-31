#! -*- coding: utf-8 -*-
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import os
import seaborn as sns
from chainer_chemistry.utils import load_json


def save_evaluation_plot(x, y, metric, filename):
    plt.figure()

    sns.set()
    ax = sns.barplot(y=x, x=y)

    for n, (label, _y) in enumerate(zip(x, y)):
        ax.annotate(
            s='{:.4g}'.format(abs(_y)),
            xy=(_y, n),
            ha='left',
            va='center',
            xytext=(5, 0),
            textcoords='offset points',
            color='gray')

    plt.title('Performance on qm9: {}'.format(metric))
    plt.xlabel(metric)
    plt.savefig(filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    args = parser.parse_args()

    x = args.methods
    y = defaultdict(list)

    for method in args.methods:
        result = load_json(os.path.join(
            args.prefix + method, 'eval_result_mae.json'))
        for label, value in result.items():
            y[label].append(value)

    for label in y.keys():
        save_evaluation_plot(
            x, y[label], label, 'eval_qm9_{}_mae.png'.format(label))


if __name__ == "__main__":
    main()
