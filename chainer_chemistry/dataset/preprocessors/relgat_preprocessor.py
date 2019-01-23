from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor  # NOQA


class RelGATPreprocessor(GGNNPreprocessor):
    """RelGCN Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.
        return_is_real_node (bool): If True, also returns `is_real_node`.

    """
    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False, return_is_real_node=True):
        super(RelGATPreprocessor, self).__init__(
            max_atoms=max_atoms, out_size=out_size, add_Hs=add_Hs,
            kekulize=kekulize, return_is_real_node=return_is_real_node)
