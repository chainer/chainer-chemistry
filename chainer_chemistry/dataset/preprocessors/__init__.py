from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.base_preprocessor import BasePreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_adj_matrix  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_atomic_number_array  # NOQA
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms  # NOQA
from chainer_chemistry.dataset.preprocessors.ecfp_preprocessor import ECFPPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.rsgcn_preprocessor import RSGCNPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.schnet_preprocessor import SchNetPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.weavenet_preprocessor import WeaveNetPreprocessor  # NOQA

preprocess_method_dict = {
    'ecfp': ECFPPreprocessor,
    'nfp': NFPPreprocessor,
    'ggnn': GGNNPreprocessor,
    'schnet': SchNetPreprocessor,
    'weavenet': WeaveNetPreprocessor,
    'rsgcn': RSGCNPreprocessor,
}
