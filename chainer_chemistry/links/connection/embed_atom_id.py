import chainer
from chainer_chemistry.config import MAX_ATOMIC_NUM


class EmbedAtomID(chainer.links.EmbedID):
    """Embeddning specialized to atoms.

    This is a chain in the sense of Chainer that converts
    an atom, represented by a sequence of molecule IDs,
    to a sequence of embedding vectors of molecules.
    The operation is done in a minibatch manner, as most chains do.

    The forward propagation of link consists of ID embedding,
    which converts the input `x` into vector embedding `h` where
    its shape represents (minibatch, atom, channel)

    .. seealso:: :class:`chainer.links.EmbedID`
    """

    def __init__(self, out_size, in_size=MAX_ATOMIC_NUM, initialW=None,
                 ignore_label=None):
        super(EmbedAtomID, self).__init__(
            in_size=in_size, out_size=out_size, initialW=initialW,
            ignore_label=ignore_label)

    def __call__(self, x):
        """Forward propagaion.

        Args:
            x (:class:`chainer.Variable`, or :class:`numpy.ndarray` \
            or :class:`cupy.ndarray`):
                Input array that should be an integer array
                whose ``ndim`` is 2. This method treats the array
                as a minibatch of atoms, each of which consists
                of a sequence of molecules represented by integer IDs.
                The first axis should be an index of atoms
                (i.e. minibatch dimension) and the second one be an
                index of molecules.

        Returns:
            :class:`chainer.Variable`:
                A 3-dimensional array consisting of embedded vectors of atoms,
                representing (minibatch, atom, channel).

        """

        h = super(EmbedAtomID, self).__call__(x)
        return h
