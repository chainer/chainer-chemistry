from chainerchem.dataset.preprocessors.common import get_atomic_numbers
from chainerchem.dataset.preprocessors.common import type_check_num_atoms
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor


class AtomicNumberPreprocessor(MolPreprocessor):
    """Atomic number Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
        number of atoms is more than this value, this data is simply
        ignored.
        Setting negative value indicates no limit for max atoms.

    """

    def __init__(self, max_atoms=-1):
        super(AtomicNumberPreprocessor, self).__init__()
        self.max_atoms = max_atoms

    def get_descriptor(self, mol):
        """get descriptor

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = get_atomic_numbers(mol, self.max_atoms)
        return atom_array
