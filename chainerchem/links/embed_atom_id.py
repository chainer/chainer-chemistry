import chainer


class EmbedAtomID(chainer.links.EmbedID):
    """Embeddning specialized to atoms.

    This is a chain in the sense of Chainer that converts
    an atom, represented by a sequence of molecule IDs,
    to a sequence of embedding vectors of molecules.
    The operation is done in a minibatch manner, as most chains do.
    
    The forward propagation of link consists of ID embedding,
    followed by transposition of second (axis=1) and third
    (axis=2) dimension.

    .. seealso:: :class:`chainer.links.EmbedID`
    """

    def __call__(self, x):
        """Forward propagaion.

        Args:
            x (:class:`chainer.Variable`, or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Input array that should be an integer array whose ``ndim`` is 2.
                This method treats the array as a minibatch of atoms, each of which consists
                of a sequence of molecules represented by integer IDs.
                The first axis should be an index of atoms
                (i.e. minibatch dimension) and the second one be an index of molecules.

        Returns:
            :class:`chainer.Variable`:
                A 3-dimensional array consisting of embedded vectors of atoms.

        """
        
        h = super(EmbedAtomID, self).__call__(x)
        return chainer.functions.transpose(h, (0, 2, 1))
