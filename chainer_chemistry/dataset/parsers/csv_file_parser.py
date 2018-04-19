from logging import getLogger

import numpy
import pandas
from rdkit import Chem
from tqdm import tqdm

from chainer_chemistry.dataset.parsers.base_parser import BaseFileParser
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

import traceback


class CSVFileParser(BaseFileParser):
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
        super(CSVFileParser, self).__init__(preprocessor)
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels  # type: list
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)

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
        logger = self.logger
        pp = self.preprocessor
        smiles_list = []

        # counter = 0
        if isinstance(pp, MolPreprocessor):
            df = pandas.read_csv(filepath)

            if target_index is not None:
                df = df.iloc[target_index]

            features = None
            smiles_index = df.columns.get_loc(self.smiles_col)
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles = row[smiles_index]
                # TODO(Nakago): Check.
                # currently it assumes list
                labels = [row[i] for i in labels_index]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        fail_count += 1
                        continue
                    # Note that smiles expression is not unique.
                    # we should re-obtain smiles from `mol`, so that the
                    # smiles order does not contradict with input features'
                    # order.
                    # Here, `smiles` and `standardized_smiles` expresses
                    # same molecule, but the expression may be different!
                    standardized_smiles, mol = pp.prepare_smiles_and_mol(mol)
                    input_features = pp.get_input_features(mol)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels = self.postprocess_label(labels)

                    if return_smiles:
                        assert standardized_smiles == Chem.MolToSmiles(mol)
                        smiles_list.append(standardized_smiles)
                        # logger.debug('[DEBUG] smiles {}, standard_smiles {}'
                        #              .format(smiles, standardized_smiles))
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'
                                   .format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    continue
                # Initialize features: list of list
                if features is None:
                    if isinstance(input_features, tuple):
                        num_features = len(input_features)
                    else:
                        num_features = 1
                    if self.labels is not None:
                        num_features += 1
                    features = [[] for _ in range(num_features)]

                if isinstance(input_features, tuple):
                    for i in range(len(input_features)):
                        features[i].append(input_features[i])
                else:
                    features[0].append(input_features)
                if self.labels is not None:
                    features[len(features) - 1].append(labels)
                success_count += 1
            ret = []

            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = numpy.empty(len(feature), dtype=numpy.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            # Spec not finalized yet for general case
            result = pp.process(filepath)

        smileses = numpy.array(smiles_list) if return_smiles else None

        if isinstance(result, tuple):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            return {"dataset": NumpyTupleDataset(*result), "smiles": smileses}
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            return {"dataset": NumpyTupleDataset(result), "smiles": smileses}

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
