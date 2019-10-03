# Network Node Classification Example

This example performs semi-supervised node classification.

## Dependencies

Before running the example, the following packages also need to be installed:

- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`scikit-learn`](http://scikit-learn.org/stable/)

## How to run the code

### Dataset

Please download the dataset to use, unzip and place it under each directory.

- [Cora](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)
- [Citeseer](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz)
- [Reddit](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip)


### Train a model

To train a model, run the following:

On the CPU:
```angular2html
PYTHONPATH=. python examples/network_graph/train_network_graph.py --dataset cora
```

On the GPU:
```angular2html
PYTHONPATH=. python examples/network_graph/train_network_graph.py --dataset cora
```
