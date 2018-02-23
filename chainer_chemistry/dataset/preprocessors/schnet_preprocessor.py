from logging import getLogger
import traceback

import numpy
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


def construct_distance_matrix(mol, out_size=-1):
    """Construct distance matrix

    Args:
        mol (Chem.Mol):
        out_size (int):

    Returns (numpy.ndarray): 2 dimensional array which represents distance
        between atoms

    """
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise MolFeatureExtractionError('out_size {} is smaller than number '
                                        'of atoms in mol {}'
                                        .format(out_size, N))

    confid = AllChem.EmbedMolecule(mol)
    try:
        dist_matrix = rdmolops.Get3DDistanceMatrix(mol, confId=confid)
    except ValueError as e:
        logger = getLogger(__name__)
        logger.info('construct_distance_matrix failed, type: {}, {}'
                    .format(type(e).__name__, e.args))
        logger.debug(traceback.format_exc())
        raise MolFeatureExtractionError

    if size > N:
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
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False):
        super(SchNetPreprocessor, self).__init__(add_Hs=add_Hs)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be equal to or larget than '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        dist_array = construct_distance_matrix(mol, out_size=self.out_size)
        return atom_array, dist_array
