import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--input-directory", "-i", default=".")
parser.add_argument("--methods", "-m", nargs="+", required=True)
parser.add_argument("--dataset", "-d", required=True)
parser.add_argument("--output-file", "-o", required=True)
args = parser.parse_args()

sns.set()

x = args.methods
y = []
for method in args.methods:
    with open(
            os.path.join(args.input_directory, "eval_" + method,
                         "eval_result.json")) as f:
        result = json.load(f)
        y.append(result["test/main/roc_auc"])

ax = sns.barplot(y=x, x=y)

for n, (label, _y) in enumerate(zip(x, y)):
    ax.annotate(
        s="{:.3f}".format(abs(_y)),
        xy=(_y, n),
        ha="right",
        va="center",
        xytext=(-5, 0),
        textcoords="offset points",
        color="white")

plt.title("Performance on {}".format(args.dataset))
plt.xlabel("ROC-AUC")
plt.savefig(args.output_file)
