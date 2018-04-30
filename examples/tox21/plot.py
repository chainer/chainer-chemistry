#! -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--names', nargs='+', required=True)
parser.add_argument('--values', nargs='+', required=True, type=float) 
args = parser.parse_args()

sns.set()

x = args.names
y = args.values

ax = sns.barplot(y=x, x=y)

for n, (label, _y) in enumerate(zip(x, y)):
    ax.annotate(
        s='{:.3f}'.format(abs(_y)),
        xy=(_y, n),
        ha='right',va='center',
        xytext=(-5, 0),
        textcoords='offset points',
        color='white'
    )
plt.title("Performance on tox21")
plt.xlabel("ROC-AUC")
plt.savefig('results.png')

