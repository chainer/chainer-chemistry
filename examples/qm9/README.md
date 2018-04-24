# QM9

QM9 dataset regression

## Dependency

This example depends on the following package as well as 
Chainer Chemistry and its dependent packages:

- [`scikit-learn`](http://scikit-learn.org/stable/)

## How to run the code

### Train the model with qm9 dataset

With CPU:
```angular2html
python train_qm9.py
```

With GPU:
```angular2html
python train_qm9.py -g 0
```

### Inference with the trained model with qm9 dataset using Regressor

As of v0.3.0, `Regressor` class is introduced which supports `predict`
method for easier inference.

`Regressor` also supports `load_pickle` method, user may load
the instance of pretrained-model using `pickle` file.

The example implemented in `predict_qm9.py`.

With CPU:
```
python predict_qm9.py [-i /path/to/training/result/directory]
```

With GPU:
```
python predict_qm9.py -g 0 [-i /path/to/training/result/directory]
```
