from logging import getLogger
import traceback

import numpy
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

from chainerchem.dataset.preprocessors.common import get_atomic_numbers
from chainerchem.dataset.preprocessors.common import type_check_num_atoms
from chainerchem.dataset.preprocessors.mol_preprocessor import MolFeatureExtractFailure  # NOQA
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor


def construct_distance_matrix(mol, zero_padding=False, num_max_atoms=-1):
    """Construct distance matrix

    Args:
        mol (Chem.Mol):
        zero_padding (bool):
        num_max_atoms (int):

    Returns:

    """
    if mol is None:
        raise MolFeatureExtractFailure('mol is None')
    N = mol.GetNumAtoms()
    size = num_max_atoms if zero_padding else N
    confid = AllChem.EmbedMolecule(mol)
    try:
        dist_matrix = rdmolops.Get3DDistanceMatrix(mol, confId=confid)
    except ValueError as e:
        logger = getLogger(__name__)
        logger.info('construct_distance_matrix failed, type: {}, {}'
                    .format(type(e).__name__, e.args))
        logger.debug(traceback.format_exc())
        raise MolFeatureExtractFailure

    if zero_padding:
        dists = numpy.zeros((size, size), dtype=numpy.float32)
        a0, a1 = dist_matrix.shape
        dists[:a0, :a1] = dist_matrix
    else:
        dists = dist_matrix
    return dists.astype(numpy.float32)


class SchNetPreprocessor(MolPreprocessor):
    """SchNet Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
        number of atoms is more than this value, this data is simply
        ignored.
        Setting negative value indicates no limit for max atoms.
        zero_padding (bool): True

    """

    def __init__(self, max_atoms=-1, zero_padding=False):
        super(SchNetPreprocessor, self).__init__()
        if zero_padding and max_atoms <= 0:
            raise ValueError('max_atoms must be set to positive value when '
                             'zero_padding is True')

        self.max_atoms = max_atoms
        self.zero_padding = zero_padding

    def get_descriptor(self, mol):
        """get descriptor

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = get_atomic_numbers(mol, self.max_atoms)
        dist_array = construct_distance_matrix(mol, self.zero_padding,
                                               num_max_atoms=self.max_atoms)
        return atom_array, dist_array
