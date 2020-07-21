# Network Node Classification Example

This example performs semi-supervised node classification.

## Dependencies

Before running the example, the following packages also need to be installed:

- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`scikit-learn`](http://scikit-learn.org/stable/)


## Supported dataset

- [Cora](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)
- [Citeseer](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz)
- [Reddit](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip)
    - we use the dataset provided by [dmlc/dgl](https://github.com/dmlc/dgl/blob/master/python/dgl/data/reddit.py) repository.

Note that dataset is downloaded automatically.

## How to run the code

### Train a model

To train a model, run the following:

On the CPU:
```angular2html
python train_network_graph.py --dataset cora
```

Train sparse model with GPU:
```angular2html
python train_network_graph.py --dataset cora --device 0 --method gin_sparse
```

### Train a model with reddit dataset

reddit dataset contains, it can run only with specific configuration.
Please turn on coo option to run training of reddit dataset.

```angular2html
python train_network_graph.py --dataset reddit --device 0 --method gin --coo true
```
