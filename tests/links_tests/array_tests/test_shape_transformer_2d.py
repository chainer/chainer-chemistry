import numpy
import pytest

from chainer_chemistry.links.array.shape_transformer_2d import ShapeTransformer2D  # NOQA


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_shape_transformer_2d_2d_array(axis):
    st = ShapeTransformer2D(axis=axis)
    x = numpy.arange(6).reshape((2, 3))
    xt = st.transform(x)
    xit = st.inverse_transform(xt)
    if axis == 0:
        assert numpy.allclose(xt.array, numpy.array([[0, 3], [1, 4], [2, 5]]))
    elif axis == 1 or axis == -1:
        assert numpy.allclose(x, xt.array)

    assert numpy.allclose(x, xit.array)


@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_shape_transformer_2d_3d_array(axis):
    st = ShapeTransformer2D(axis=axis)
    x = numpy.arange(12).reshape((2, 3, 2))
    xt = st.transform(x)
    xit = st.inverse_transform(xt)
    if axis == 0:
        assert numpy.allclose(
            xt.array,
            numpy.array([[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]]))
    elif axis == 1:
        assert numpy.allclose(
            xt.array,
            numpy.array([[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]]))
    elif axis == 2 or axis == -1:
        assert numpy.allclose(
            xt.array, x.reshape(6, 2))
    assert numpy.allclose(x, xit.array)


def test_shape_transformer_2d_error():
    st = ShapeTransformer2D(axis=1)
    x = numpy.arange(6).reshape(2, 3)
    with pytest.raises(AttributeError):
        # call `inverse_transform` before `transform`
        xt = st.inverse_transform(x)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
