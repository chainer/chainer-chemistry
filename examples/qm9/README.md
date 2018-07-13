# QM9 Regression Example

This example performs regression on the QM9 dataset.

## Dependencies

To run the example, the following packages also need to be installed:

- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`scikit-learn`](http://scikit-learn.org/stable/)

## How to run the code

### Training a model

To train a model, run the following:

On the CPU:
```angular2html
python train_qm9.py
```

On the GPU:
```angular2html
python train_qm9.py -g 0
```

### Inference using a pretrained model

As of v0.3.0, the `Regressor` class has been introduced, which provides the
`predict` method for easier inference. `Regressor` also supports the
`load_pickle` method, which allows for loading of a pretrained model, using the
`pickle` library.

The perform inference using a pretrained model, run the following:

On the CPU:
```
python predict_qm9.py [-i /path/to/training/result/directory]
```

On the GPU:
```
python predict_qm9.py -g 0 [-i /path/to/training/result/directory]
```
