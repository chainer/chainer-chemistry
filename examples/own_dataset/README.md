# Example of using your own dataset
## Usage
```
python train.py dataset.csv --label value1 value2
```

## How to use your own dataset
1. Prepare a CSV file which contains the list of SMILES and the values you want to train.
The first line of the CSV file should be label names.
See `dataset.csv` as an example.

2. Use `CSVFileParser` of Cheiner Chemistry to feed data to model.
See `train.csv` as an example.
