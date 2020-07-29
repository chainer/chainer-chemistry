from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.base_preprocessor import BasePreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.cgcnn_preprocessor import CGCNNPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_adj_matrix  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_atomic_number_array  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_discrete_edge_matrix  # NOQA
from chainer_chemistry.dataset.preprocessors.common import construct_supernode_feature  # NOQA
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms  # NOQA
from chainer_chemistry.dataset.preprocessors.ecfp_preprocessor import ECFPPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gin_preprocessor import GINPreprocessor, GINSparsePreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gnnfilm_preprocessor import GNNFiLMPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gwm_preprocessor import GGNNGWMPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gwm_preprocessor import GINGWMPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gwm_preprocessor import NFPGWMPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gwm_preprocessor import RSGCNGWMPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.megnet_preprocessor import MEGNetPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.relgat_preprocessor import RelGATPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.relgcn_preprocessor import RelGCNPreprocessor, RelGCNSparsePreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.rsgcn_preprocessor import RSGCNPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.schnet_preprocessor import SchNetPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.weavenet_preprocessor import WeaveNetPreprocessor  # NOQA

preprocess_method_dict = {
    'ecfp': ECFPPreprocessor,
    'nfp': NFPPreprocessor,
    'nfp_gwm': NFPGWMPreprocessor,
    'ggnn': GGNNPreprocessor,
    'ggnn_gwm': GGNNGWMPreprocessor,
    'gin': GINPreprocessor,
    'gin_gwm': GINGWMPreprocessor,
    'schnet': SchNetPreprocessor,
    'weavenet': WeaveNetPreprocessor,
    'relgcn': RelGCNPreprocessor,
    'rsgcn': RSGCNPreprocessor,
    'rsgcn_gwm': RSGCNGWMPreprocessor,
    'relgat': RelGATPreprocessor,
    'relgcn_sparse': RelGCNSparsePreprocessor,
    'gin_sparse': GINSparsePreprocessor,
    'gnnfilm': GNNFiLMPreprocessor,
    'megnet': MEGNetPreprocessor,
    'cgcnn': CGCNNPreprocessor
}
