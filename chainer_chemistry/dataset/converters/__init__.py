from chainer_chemistry.dataset.converters.cgcnn_converter import cgcnn_converter  # NOQA
from chainer_chemistry.dataset.converters.concat_mols import concat_mols  # NOQA
from chainer_chemistry.dataset.converters.megnet_converter import megnet_converter  # NOQA

converter_method_dict = {
    'ecfp': concat_mols,
    'nfp': concat_mols,
    'nfp_gwm': concat_mols,
    'ggnn': concat_mols,
    'ggnn_gwm': concat_mols,
    'gin': concat_mols,
    'gin_gwm': concat_mols,
    'schnet': concat_mols,
    'weavenet': concat_mols,
    'relgcn': concat_mols,
    'rsgcn': concat_mols,
    'rsgcn_gwm': concat_mols,
    'relgat': concat_mols,
    'gnnfilm': concat_mols,
    'megnet': megnet_converter,
    'cgcnn': cgcnn_converter
}
