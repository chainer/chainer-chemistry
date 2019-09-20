import chainer
from chainer_chemistry.dataset.graph_dataset.base_graph_data import PaddingGraphData  # NOQA


class PaddingModelWrapper(chainer.Chain):
    def __init__(self, predictor):
        super(PaddingModelWrapper, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def forward(self, data):
        assert isinstance(data, PaddingGraphData)
        return self.predictor(data.x, data.adj)
