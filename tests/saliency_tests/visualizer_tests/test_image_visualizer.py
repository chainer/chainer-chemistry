import os

import numpy
import pytest


from chainer_chemistry.saliency.visualizer.image_visualizer import ImageVisualizer  # NOQA


def test_image_visualizer(tmpdir):
    # Only test file is saved without error
    ch = 3
    h = 32
    w = 32
    saliency = numpy.random.uniform(0, 1, (ch, h, w))
    visualizer = ImageVisualizer()

    # 1. test with setting save_filepath
    save_filepath = os.path.join(str(tmpdir), 'tmp.png')
    visualizer.visualize(saliency, save_filepath=save_filepath)
    assert os.path.exists(save_filepath)

    # 2. test with `save_filepath=None` runs without error
    image = numpy.random.uniform(0, 1, (ch, h, w))
    visualizer.visualize(
        saliency, save_filepath=None, image=image, show_colorbar=True)


def test_table_visualizer_assert_raises():
    visualizer = ImageVisualizer()
    with pytest.raises(ValueError):
        # --- Invalid saliency shape ---
        saliency_invalid = numpy.array([0.5, 0.3, 0.2])
        visualizer.visualize(saliency_invalid)

    ch = 3
    h = 32
    w = 32
    saliency = numpy.random.uniform(0, 1, (ch, h, w))

    with pytest.raises(ValueError):
        # --- Invalid sort key ---
        image_invalid = numpy.array([0.5, 0.3, 0.2])
        visualizer.visualize(saliency, image=image_invalid)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
