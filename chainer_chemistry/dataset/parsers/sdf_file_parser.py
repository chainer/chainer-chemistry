from logging import getLogger

import numpy
from rdkit import Chem
from tqdm import tqdm

from chainer_chemistry.dataset.parsers.base_parser import BaseFileParser
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


class SDFFileParser(BaseFileParser):
    """sdf file parser

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        labels (str or list): labels column
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor, labels=None, postprocess_label=None,
                 postprocess_fn=None, logger=None):
        super(SDFFileParser, self).__init__(preprocessor)
        self.labels = labels
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)

    def parse(self, filepath, return_smiles=False, target_index=None):
        """parse sdf file using `preprocessor`

        Note that label is extracted from preprocessor's method.

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

        if isinstance(pp, MolPreprocessor):
            mol_supplier = Chem.SDMolSupplier(filepath)

            if target_index is None:
                target_index = list(range(len(mol_supplier)))

            features = None

            total_count = len(mol_supplier)
            fail_count = 0
            success_count = 0
            for index in tqdm(target_index):
                # `mol_supplier` does not accept numpy.integer, we must use int
                mol = mol_supplier[int(index)]

                if mol is None:
                    total_count -= 1
                    continue
                try:
                    # Labels need to be extracted from `mol` before standardize
                    # smiles.
                    if self.labels is not None:
                        label = pp.get_label(mol, self.labels)
                        if self.postprocess_label is not None:
                            label = self.postprocess_label(label)

                    # Note that smiles expression is not unique.
                    # we should re-obtain smiles from `mol`, so that the
                    # smiles order does not contradict with input features'
                    # order.
                    # Here, `smiles` and `standardized_smiles` expresses
                    # same molecule, but the expression may be different!
                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    standardized_smiles, mol = pp.prepare_smiles_and_mol(mol)
                    input_features = pp.get_input_features(mol)

                    # Initialize features: list of list
                    if features is None:
                        if isinstance(input_features, tuple):
                            num_features = len(input_features)
                        else:
                            num_features = 1
                        if self.labels is not None:
                            num_features += 1
                        features = [[] for _ in range(num_features)]

                    if return_smiles:
                        assert standardized_smiles == Chem.MolToSmiles(mol)
                        smiles_list.append(standardized_smiles)
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    continue
                except Exception as e:
                    logger.warning('parse() error, type: {}, {}'
                                   .format(type(e).__name__, e.args))
                    continue

                if isinstance(input_features, tuple):
                    for i in range(len(input_features)):
                        features[i].append(input_features[i])
                else:
                    features[0].append(input_features)
                if self.labels is not None:
                    features[len(features) - 1].append(label)
                success_count += 1

            ret = []

            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around to convert object-type list into
                    # numpy array.
                    # See, https://goo.gl/kgJXwb
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
                result = NumpyTupleDataset(result)
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
        mol_supplier = Chem.SDMolSupplier(filepath)
        return len(mol_supplier)
