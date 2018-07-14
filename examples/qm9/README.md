# QM9 Regression Example

This example performs regression on the QM9 dataset.

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

### Evaluation of implemented models

To evaluate the performance of the currently implemented models, run the
following:

On the CPU:
```
bash evaluate_models_qm9.sh [epoch]
```

On the GPU:
```
bash evaluate_models_qm9.sh [epoch] 0
```

This scripts start the training process for a number of `epoch` epochs per
model. Inference is then performed and evaluation metrics are reported. For
regression tasks (such as with QM9), these are MAE and RMSE. One plot per
metric is then createad (saved as `eval_[metric]_qm9.png` in the example
directory), which outputs these values as reported by the diffent models.
