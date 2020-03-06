import warnings

from chainer_chemistry import dataset  # NOQA
try:
    from chainer_chemistry import datasets  # NOQA
except ImportError as e:
    if 'rdkit' in e.msg:
        warnings.warn(
            'A module chainer_chemistry.datasets was not imported, '
            'probably because RDKit is not installed. '
            'To install RDKit, please follow instruction in '
            'https://github.com/pfnet-research/chainer-chemistry#installation.',  # NOQA
            UserWarning)
    else:
        raise(e)
from chainer_chemistry import functions  # NOQA
from chainer_chemistry import links  # NOQA
from chainer_chemistry import models  # NOQA
from chainer_chemistry import training  # NOQA

# --- config variable definitions ---
from chainer_chemistry.config import *  # NOQA


from chainer_chemistry import _version  # NOQA


__version__ = _version.__version__
