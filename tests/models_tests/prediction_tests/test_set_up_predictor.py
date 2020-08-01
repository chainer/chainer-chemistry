from typing import Dict  # NOQA

import chainer  # NOQA
import pytest

from chainer_chemistry.models.ggnn import GGNN
from chainer_chemistry.models.gin import GIN
from chainer_chemistry.models.gnn_film import GNNFiLM
from chainer_chemistry.models.nfp import NFP
from chainer_chemistry.models.prediction.graph_conv_predictor import GraphConvPredictor  # NOQA
from chainer_chemistry.models.prediction.set_up_predictor import set_up_predictor  # NOQA
from chainer_chemistry.models.relgat import RelGAT
from chainer_chemistry.models.relgcn import RelGCN
from chainer_chemistry.models.rsgcn import RSGCN
from chainer_chemistry.models.schnet import SchNet
from chainer_chemistry.models.weavenet import WeaveNet

from chainer_chemistry.models.gwm.gwm_net import GGNN_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import GIN_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import NFP_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import RSGCN_GWM  # NOQA

from chainer_chemistry.models.cwle.cwle_net import GGNN_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import RelGAT_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import RelGCN_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import GIN_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import NFP_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import RSGCN_CWLE  # NOQA

from chainer_chemistry.models.gwle.gwle_net import GGNN_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import RelGAT_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import RelGCN_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import GIN_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import NFP_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import RSGCN_GWLE  # NOQA


class_num = 7
n_unit = 11
conv_layers = 3


@pytest.fixture
def models_dict():
    # type: () -> Dict[str, chainer.Link]
    return {
        'nfp': NFP,
        'ggnn': GGNN,
        'schnet': SchNet,
        'weavenet': WeaveNet,
        'rsgcn': RSGCN,
        'relgcn': RelGCN,
        'relgat': RelGAT,
        'gin': GIN,
        'nfp_gwm': NFP_GWM,
        'ggnn_gwm': GGNN_GWM,
        'rsgcn_gwm': RSGCN_GWM,
        'gin_gwm': GIN_GWM,
        'gnnfilm': GNNFiLM,
        'nfp_wle': NFP,
        'ggnn_wle': GGNN,
        'relgat_wle': RelGAT,
        'relgcn_wle': RelGCN,
        'rsgcn_wle': RSGCN,
        'gin_wle': GIN,
        'nfp_cwle': NFP_CWLE,
        'ggnn_cwle': GGNN_CWLE,
        'relgat_cwle': RelGAT_CWLE,
        'relgcn_cwle': RelGCN_CWLE,
        'rsgcn_cwle': RSGCN_CWLE,
        'gin_cwle': GIN_CWLE,
        'nfp_gwle': NFP_GWLE,
        'ggnn_gwle': GGNN_GWLE,
        'relgat_gwle': RelGAT_GWLE,
        'relgcn_gwle': RelGCN_GWLE,
        'rsgcn_gwle': RSGCN_GWLE,
        'gin_gwle': GIN_GWLE
    }


def test_setup_predictor(models_dict):
    # type: (Dict[str, chainer.Link]) -> None
    for method, instance in models_dict.items():
        predictor = set_up_predictor(
            method=method,
            n_unit=n_unit,
            conv_layers=conv_layers,
            class_num=class_num)
        assert isinstance(predictor.graph_conv, instance)
        assert isinstance(predictor, GraphConvPredictor)


def test_call_invalid_model():
    # type: () -> None
    with pytest.raises(ValueError):
        set_up_predictor(
            method='invalid',
            n_unit=n_unit,
            conv_layers=conv_layers,
            class_num=class_num)


def test_set_up_predictor_with_conv_kwargs():
    # type: () -> None
    predictor = set_up_predictor(
        method='nfp',
        n_unit=n_unit,
        conv_layers=conv_layers,
        class_num=class_num,
        conv_kwargs={
            'max_degree': 4,
            'concat_hidden': True
        })
    assert predictor.graph_conv.max_degree == 4
    assert predictor.graph_conv.concat_hidden is True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
