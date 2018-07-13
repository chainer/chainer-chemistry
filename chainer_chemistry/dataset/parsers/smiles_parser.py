import pandas

from chainer_chemistry.dataset.parsers.data_frame_parser import DataFrameParser


class SmilesParser(DataFrameParser):
    """smiles parser

    It parses `smiles_list`, which is a list of string of smiles.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor,
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(SmilesParser, self).__init__(
            preprocessor, labels=None, smiles_col='smiles',
            postprocess_label=postprocess_label, postprocess_fn=postprocess_fn,
            logger=logger)

    def parse(self, smiles_list, return_smiles=False, target_index=None,
              return_is_successful=False):
        """parse `smiles_list` using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            smiles_list (list): list of strings of smiles
            return_smiles (bool): If set to True, this function returns
                preprocessed dataset and smiles list.
                If set to False, this function returns preprocessed dataset and
                `None`.
            target_index (list or None): target index list to partially extract
                dataset. If None (default), all examples are parsed.
            return_is_successful (bool): If set to `True`, boolean list is
                returned in the key 'is_successful'. It represents
                preprocessing has succeeded or not for each SMILES.
                If set to False, `None` is returned in the key 'is_success'.

        Returns (dict): dictionary that contains Dataset, 1-d numpy array with
            dtype=object(string) which is a vector of smiles for each example
            or None.

        """
        df = pandas.DataFrame({'smiles': smiles_list})
        return super(SmilesParser, self).parse(
            df, return_smiles=return_smiles, target_index=target_index,
            return_is_successful=return_is_successful)

    def extract_total_num(self, smiles_list):
        """Extracts total number of data which can be parsed

        We can use this method to determine the value fed to `target_index`
        option of `parse` method. For example, if we want to extract input
        feature from 10% of whole dataset, we need to know how many samples
        are in a file. The returned value of this method may not to be same as
        the final dataset size.

        Args:
            smiles_list (list): list of strings of smiles

        Returns (int): total number of dataset can be parsed.

        """
        return len(smiles_list)
