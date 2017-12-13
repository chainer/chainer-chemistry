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
        filepath (str) : file path of csv file
        preprocessor (BasePreprocessor):
        labels (str or list): labels column
        smiles_col (str): smiles column
        mol (str):
    """

    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col='smiles',
                 postprocess_label=None, postprocess_fn=None):
        super(CSVFileParser, self).__init__(preprocessor)
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels  # type: list
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.smiles = None

    def parse(self, filepath, retain_smiles=False):
        """parse csv file using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            filepath (str): file path to be parsed.
            retain_smiles (bool): If set to True, smiles list is saved to
                `smiles` property.

        Returns: Dataset

        """
        logger = getLogger(__name__)
        pp = self.preprocessor
        if retain_smiles:
            self.smiles = []  # Initialize

        # counter = 0
        if isinstance(pp, MolPreprocessor):
            df = pandas.DataFrame.from_csv(filepath)

            features = None
            smiles_index = df.columns.get_loc(self.smiles_col)
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

                    if retain_smiles:
                        assert standardized_smiles == Chem.MolToSmiles(mol)
                        self.smiles.append(standardized_smiles)
                        # logger.debug('[DEBUG] smiles {}, standard_smiles {}'
                        #              .format(smiles, standardized_smiles))
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    continue
                except Exception as e:
                    logger = getLogger(__name__)
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

        if isinstance(result, tuple):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            return NumpyTupleDataset(*result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            return NumpyTupleDataset(result)
