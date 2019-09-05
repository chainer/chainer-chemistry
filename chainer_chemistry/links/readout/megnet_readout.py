import chainer
from chainer import functions


from chainer_chemistry.links.readout.set2set import Set2Set


class MEGNetReadout(chainer.Chain):
    """MEGNet submodule for readout part.

    Args:
        in_channels (int): dimension of feature vector associated to
            each node. Must not be `None`.
        n_layers (int): number of LSTM layers for set2set
        processing_steps (int): number of processing for set2set
    """

    def __init__(self, in_channels=32, n_layers=16, processing_steps=3):
        super(MEGNetReadout, self).__init__()
        if processing_steps <= 0:
            raise ValueError("[ERROR] Unexpected value processing_steps={}"
                             .format(processing_steps))

        with self.init_scope():
            self.set2set = Set2Set(in_channels=in_channels, n_layers=n_layers)
        self.processing_steps = processing_steps

    def __call__(self, h, **kwargs):
        self.set2set.reset_state()
        # TODO: check the thesis
        for i in range(self.processing_steps):
            g = self.set2set(h)
        return g
