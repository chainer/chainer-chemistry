# Material Project Regression Example

This example performs regression on the Material Project dataset.

## Dependencies

Before running the example, the following packages also need to be installed:

- [`matplotlib`](https://matplotlib.org/)
- [`scikit-learn`](http://scikit-learn.org/stable/)

## How to run the code

### Train a model

To train a model, run the following:

On the CPU:
```angular2html
python train_mp.py
```

On the GPU:
```angular2html
python train_mp.py -g 0
```

### Inference using a pretrained model

As of v0.3.0, the `Regressor` class has been introduced, which provides the
`predict` method for easier inference. `Regressor` also supports the
`load_pickle` method, which allows for loading of a pretrained model, using the
`pickle` library.

The perform inference using a pretrained model, run the following:

On the CPU:
```
python predict_mp.py [-i /path/to/training/result/directory]
```

On the GPU:
```
python predict_mp.py -g 0 [-i /path/to/training/result/directory]
```
