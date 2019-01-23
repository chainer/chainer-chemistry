from chainer_chemistry.dataset.preprocessors.common import \
    construct_adj_matrix, construct_is_real_node
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


class NFPPreprocessor(MolPreprocessor):
    """NFP Preprocessor

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
        super(NFPPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.return_is_real_node = return_is_real_node

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:
            atom_array (numpy.ndarray): (node,)
            adj_array (numpy.ndarray): (node, node)
            is_real_node (numpy.ndarray): (node,)

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_adj_matrix(mol, out_size=self.out_size)
        if not self.return_is_real_node:
            return atom_array, adj_array
        else:
            is_real_node = construct_is_real_node(
                mol, out_size=self.out_size)
            return atom_array, adj_array, is_real_node
