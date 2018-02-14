import chainer


class GraphBatchNormalization(chainer.links.BatchNormalization):
    """Graph Batch Normalization layer.

    .. seealso:: :class:`chainer.links.BatchNormalization`
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

        # The implemenataion of batch normalization for graph convolution below
        # is rather naive. To be precise, it is necessary to consider the
        # difference in the number of atoms for each graph. However, the
        # implementation below does not take it into account, and assumes
        # that all graphs have the same number of atoms, hence extra numbers
        # of zero are included when average is computed. In other word, the
        # results of batch normalization below is biased.

        s0, s1, s2 = h.shape
        h = chainer.functions.reshape(h, (s0 * s1, s2))
        h = super(GraphBatchNormalization, self).__call__(h)
        h = chainer.functions.reshape(h, (s0, s1, s2))

        return h
