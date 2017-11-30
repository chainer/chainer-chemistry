from chainerchem.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainerchem.dataset.preprocessors.base_preprocessor import BasePreprocessor  # NOQA
from chainerchem.dataset.preprocessors.common import construct_atomic_number_array  # NOQA
from chainerchem.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainerchem.dataset.preprocessors.common import type_check_num_atoms  # NOQA
from chainerchem.dataset.preprocessors.ecfp_preprocessor import ECFPPreprocessor  # NOQA
from chainerchem.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor  # NOQA
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainerchem.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor  # NOQA
from chainerchem.dataset.preprocessors.schnet_preprocessor import SchNetPreprocessor  # NOQA
from chainerchem.dataset.preprocessors.weavenet_preprocessor import WeaveNetPreprocessor  # NOQA

preprocess_method_dict = {
    'ecfp': ECFPPreprocessor,
    'nfp': NFPPreprocessor,
    'ggnn': GGNNPreprocessor,
    'schnet': SchNetPreprocessor,
    'weavenet': WeaveNetPreprocessor,
}
