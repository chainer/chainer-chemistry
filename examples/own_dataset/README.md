# Example of using your own dataset

This example shows how to train models with your own dataset stored in the CSV format.

A regression task is performed using [`Regressor`](http://chainer-chemistry.readthedocs.io/en/stable/generated/chainer_chemistry.models.Regressor.html#chainer_chemistry.models.Regressor). For a classification setting that makes use of [`Classifier`](http://chainer-chemistry.readthedocs.io/en/stable/generated/chainer_chemistry.models.Classifier.html#chainer_chemistry.models.Classifier), 
please refer to the `tox21` example.

## Dependencies

Before running the example, the following packages also need to be installed:

- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`scikit-learn`](http://scikit-learn.org/stable/)

## How to run the code

### Dataset preparation

Prepare a CSV file containing the training data samples, one per row. Each row contains the SMILES string of one molecule, followed by the (label) values of the molecule's desired properties. The first line of the CSV file contains label names.

Below you can find an example:

```
SMILES,value1,value2
CC1CC1CN1CC1C,-0.2190999984741211,0.08590000122785568
C#CCC(=N)OC=O,-0.2750999927520752,-0.032999999821186066
Cc1cnc(C=O)n1C,-0.23080000281333923,-0.053700000047683716
N=COCC(C=O)CO,-0.26260000467300415,-0.043699998408555984
[...]
```

Save one CSV file for training (e.g., `dataset_train.csv`) and one for testing (e.g., `dataset_test.csv`). Then pass them to the training and testing scripts, as shown below.

### Train a model

To train a new model, run the following:
```
python train_own_dataset.py --datafile dataset_train.csv --label value1 value2
```

The `--label` option specifies which columns in `dataset_train.csv` are trained.
Type `python train_own_dataset.py --help` to see the complete set of options.

### Inference using a pretrained model

To perform inference using a pretrained model, run the following:
```
python predict_own_dataset.py --datafile dataset_test.csv --label value1 value2
```
Type `python test_own_dataset.py --help` to see the complete set of options.

### Evaluation of implemented models

To evaluate the performance of the currently implemented models, run the following:
```
bash evaluate_own_dataset.sh [gpu_id] [epoch]
```
where `gpu_id` is the identifier of your GPU and `epoch` is the number of training epochs.
To run the code on CPU, set `gpu_id` to `-1`.

The scripts start the training process. Inference is then performed and evaluation metrics are reported. 
For regression tasks (such as the current example), these are MAE and RMSE. 
One plot per metric is created (saved as `eval_[metric]_own.png` in the example directory), which outputs these values as reported by the different models.
