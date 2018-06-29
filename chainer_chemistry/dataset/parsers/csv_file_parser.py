import pandas

from chainer_chemistry.dataset.parsers.data_frame_parser import DataFrameParser


class CSVFileParser(DataFrameParser):
    """csv file parser

    This FileParser parses .csv file.
    It should contain column which contain SMILES as input, and
    label column which is the target to predict.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        labels (str or list): labels column
        smiles_col (str): smiles column
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col='smiles',
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(DataFrameParser, self).__init__(preprocessor, labels, smiles_col,
                                              postprocess_label,
                                              postprocess_fn, logger)

    def parse(self, filepath, return_smiles=False, target_index=None):
        """parse csv file using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            filepath (str): file path to be parsed.
            return_smiles (bool): If set to True, this function returns
                preprocessed dataset and smiles list.
                If set to False, this function returns preprocessed dataset and
                `None`.
            target_index (list or None): target index list to partially extract
                dataset. If None (default), all examples are parsed.

        Returns (dict): dictionary that contains Dataset, 1-d numpy array with
            dtype=object(string) which is a vector of smiles for each example
            or None.

        """
        df = pandas.read_csv(filepath)
        return super(DataFrameParser, self).parse(df, return_smiles,
                                                  target_index)

    def extract_total_num(self, filepath):
        """Extracts total number of data which can be parsed

        We can use this method to determine the value fed to `target_index`
        option of `parse` method. For example, if we want to extract input
        feature from 10% of whole dataset, we need to know how many samples
        are in a file. The returned value of this method may not to be same as
        the final dataset size.

        Args:
            filepath (str): file path of to check the total number.

        Returns (int): total number of dataset can be parsed.

        """
        df = pandas.read_csv(filepath)
        return len(df)
