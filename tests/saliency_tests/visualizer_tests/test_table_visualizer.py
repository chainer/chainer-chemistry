import os

import numpy
import pytest


from chainer_chemistry.saliency.visualizer.table_visualizer import TableVisualizer  # NOQA


def test_table_visualizer(tmpdir):
    # Only test file is saved without error
    saliency = numpy.array([0.5, 0.3, 0.2])
    visualizer = TableVisualizer()

    # 1. test with setting save_filepath
    save_filepath = os.path.join(str(tmpdir), 'tmp.png')
    visualizer.visualize(saliency, save_filepath=save_filepath)
    assert os.path.exists(save_filepath)
    # 2. test with `save_filepath=None` runs without error
    visualizer.visualize(
        saliency, save_filepath=None, feature_names=['hoge', 'huga', 'piyo'],
        num_visualize=2)


def test_table_visualizer_assert_raises():
    visualizer = TableVisualizer()
    with pytest.raises(ValueError):
        # --- Invalid saliency shape ---
        saliency = numpy.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        visualizer.visualize(saliency)

    with pytest.raises(ValueError):
        # --- Invalid sort key ---
        saliency = numpy.array([0.5, 0.3, 0.2])
        visualizer.visualize(saliency, sort='invalidkey')

    with pytest.raises(ValueError):
        # --- Invalid feature_names key ---
        saliency = numpy.array([0.5, 0.3, 0.2])
        feature_names = ['a', 'b', 'c', 'd']
        visualizer.visualize(saliency, feature_names=feature_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
