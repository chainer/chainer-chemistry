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
        filepath:
        preprocessor:
        labels (str or list): labels column
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
    """

    def __init__(self, preprocessor, labels=None, postprocess_label=None,
                 postprocess_fn=None):
        super(SDFFileParser, self).__init__(preprocessor)
        self.labels = labels
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.smiles = None

    def parse(self, filepath, retain_smiles=False):
        """parse sdf file using `preprocessor`

        Note that label is extracted from preprocessor's method.

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

        if isinstance(pp, MolPreprocessor):
            mol_supplier = Chem.SDMolSupplier(filepath)

            features = None

            total_count = len(mol_supplier)
            fail_count = 0
            success_count = 0
            for mol in tqdm(mol_supplier):
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

                    if retain_smiles:
                        assert standardized_smiles == Chem.MolToSmiles(mol)
                        self.smiles.append(standardized_smiles)
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    continue
                except Exception as e:
                    logger = getLogger(__name__)
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

        if isinstance(result, tuple):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            return NumpyTupleDataset(*result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            return NumpyTupleDataset(result)
