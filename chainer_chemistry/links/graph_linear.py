import chainer


class GraphLinear(chainer.links.Linear):
    """Graph Linear layer.

    This function assumes its input is 3-dimensional.
    Differently from :class:`chainer.functions.linear`, it applies an affine
    transformation to the third axis of input `x`.

    .. seealso:: :class:`chainer.links.Linear`
    """

    def __call__(self, x):
        """Forward propagation.

        Args:
            x (:class:`chainer.Variable`, or :class:`numpy.ndarray`\
            or :class:`cupy.ndarray`):
                Input array that should be a float array whose ``ndim`` is 3.

                It represents a minibatch of atoms, each of which consists
                of a sequence of molecules. Each molecule is represented
                by integer IDs. The first axis is an index of atoms
                (i.e. minibatch dimension) and the second one an index
                of molecules.

        Returns:
            :class:`chainer.Variable`:
                A 3-dimeisional array.

        """
        h = x
        # (minibatch, atom, ch)
        s0, s1, s2 = h.shape
        h = chainer.functions.reshape(h, (s0 * s1, s2))
        h = super(GraphLinear, self).__call__(h)
        h = chainer.functions.reshape(h, (s0, s1, self.out_size))
        return h
