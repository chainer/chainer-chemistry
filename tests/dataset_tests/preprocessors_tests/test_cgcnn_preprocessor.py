import pytest

from chainer_chemistry.dataset.preprocessors import CGCNNPreprocessor


def test_cgcnn_preprocessor_init():
    pp = CGCNNPreprocessor()
    print('pp.atom_features', pp.atom_features)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
