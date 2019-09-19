# Network Node Classification Example

This example performs semi-supervised node classification.

## Dependencies

Before running the example, the following packages also need to be installed:

- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`scikit-learn`](http://scikit-learn.org/stable/)

## How to run the code

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
