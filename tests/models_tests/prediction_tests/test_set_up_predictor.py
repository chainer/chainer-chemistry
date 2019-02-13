from typing import Dict  # NOQA
from typing import Tuple  # NOQA

import chainer  # NOQA
import pytest

from chainer_chemistry.models.ggnn import GGNN
from chainer_chemistry.models.nfp import NFP
from chainer_chemistry.models.prediction.graph_conv_predictor import GraphConvPredictor  # NOQA
from chainer_chemistry.models.prediction.set_up_predictor import set_up_predictor  # NOQA
from chainer_chemistry.models.relgat import RelGAT
from chainer_chemistry.models.relgcn import RelGCN
from chainer_chemistry.models.rsgcn import RSGCN
from chainer_chemistry.models.schnet import SchNet
from chainer_chemistry.models.weavenet import WeaveNet

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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
