from chainer_chemistry.dataset.splitters import base_splitter  # NOQA
from chainer_chemistry.dataset.splitters import random_splitter  # NOQA
from chainer_chemistry.dataset.splitters import scaffold_splitter  # NOQA
from chainer_chemistry.dataset.splitters import stratified_splitter  # NOQA

from chainer_chemistry.dataset.splitters.base_splitter import BaseSplitter  # NOQA
from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter  # NOQA
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter  # NOQA
from chainer_chemistry.dataset.splitters.stratified_splitter import StratifiedSplitter  # NOQA

split_method_dict = {
    'random': RandomSplitter,
    'stratified': StratifiedSplitter,
    'scaffold': ScaffoldSplitter,
}
