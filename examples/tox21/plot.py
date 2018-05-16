#! -*- coding: utf-8 -*-
import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', required=True)
parser.add_argument('--methods', nargs='+', required=True)
args = parser.parse_args()

sns.set()

x = args.methods
y = []
for method in args.methods:
    with open(os.path.join(args.prefix + method, 'eval_result.json')) as f:
        result = json.load(f)
        y.append(result["test/main/roc_auc"])

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
plt.savefig('eval_results_tox21.png')

