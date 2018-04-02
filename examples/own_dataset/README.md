# Example of using your own dataset
## Usage
```
python train.py dataset.csv --label value1 value2
```

The `--label` option specifies which columns in `dataset.csv` are trained.
Type `python train.py --help` to see complete options.

## Procedure
1. Prepare a CSV file which contains a list of SMILES and values you want to train.
The first line of the CSV file should be label names.
See `dataset.csv` as an example.
`dataset.csv` is made by sampling from the QM9 dataset.
`value1` is HOMO and `value2` is LUMO.

2. Use [CSVFileParser](http://chainer-chemistry.readthedocs.io/en/stable/generated/chainer_chemistry.dataset.parsers.CSVFileParser.html) of Cheiner Chemistry to feed data to model.
See `train.csv` as an example.
