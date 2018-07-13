# Example of using your own dataset

This example shows how to train models with your own dataset stored in CSV format.

We will solve a regression task with [`Regressor`](http://chainer-chemistry.readthedocs.io/en/stable/generated/chainer_chemistry.models.Regressor.html#chainer_chemistry.models.Regressor).
For a classification setting that makes use of [`Classifier`](http://chainer-chemistry.readthedocs.io/en/stable/generated/chainer_chemistry.models.Classifier.html#chainer_chemistry.models.Classifier), 
please refer to the tox21 example.

## Usage
```
python train_custom_dataset.py --datafile dataset_train.csv --label value1 value2
```

The `--label` option specifies which columns in `dataset_train.csv` are trained.
Type `python train.py --help` to see complete options.

## Procedure
1. Prepare a CSV file which contains a list of SMILES and values you want to train.
The first line of the CSV file should be label names.
See `dataset_train.csv` as an example.
`dataset_train.csv` and `dataset_test.csv` are made by sampling from the QM9 dataset.
`value1` is HOMO and `value2` is LUMO.

2. Use [CSVFileParser](http://chainer-chemistry.readthedocs.io/en/stable/generated/chainer_chemistry.dataset.parsers.CSVFileParser.html) of Cheiner Chemistry to feed data to model.
See `train.csv` as an example.
