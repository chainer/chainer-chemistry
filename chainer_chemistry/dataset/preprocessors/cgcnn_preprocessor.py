import numpy
import json


from chainer_chemistry.dataset.utils import GaussianDistance
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


class CGCNNPreprocessor(MolPreprocessor):
    """CGCNNPreprocessor

    Args:
    TODO: For Molecule

    For Crystal
        max_neighbors (int): Max number of atom considered as neighbors
        max_radius (float): Cutoff radius (angstrom)
        exapand_dim (int): This value is equal to pair feature dimension
    """

    def __init__(self, max_neighbors=12, max_radius=8, exapand_dim=40):
        super(CGCNNPreprocessor, self).__init__()

        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.exapand_dim = exapand_dim
        self.gdf = GaussianDistance(centers=numpy.linspace(0, 8, exapand_dim))

    def get_input_feature_from_crystal(self, crystal):
        """get input features from crystal object

        Args:
            crystal (Crystal):

        """

        # get atom feature vector
        # TODO: ここも適当なPATHに置き換える
        path = '/Users/i19_nissy/code/github.com/nd-02110114/' + \
            'chainer-chemistry/examples/mp/assets/atom_init.json'
        feat_dict = json.load(open(path))
        initial_atom_features = {int(key): numpy.array(value,
                                                       dtype=numpy.float32)
                                 for key, value in feat_dict.items()}
        atom_feature = numpy.vstack(
            [initial_atom_features[crystal[i].specie.number]
             for i in range(len(crystal))]
        )

        # get edge feture vector
        neighbor_indexes = list()
        neighbor_features = list()
        all_neighbors = crystal.get_all_neighbors(
            self.max_radius, include_index=True)
        all_neighbors = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbors]  # NOQA

        for nbrs in all_neighbors:
            nbr_feature_idx = numpy.zeros(
                self.max_neighbors, dtype=numpy.int32)
            nbr_feature = numpy.zeros(
                self.max_neighbors, dtype=numpy.float32) + self.max_radius + 1.
            nbr_feature_idx[:len(nbrs)] = list(
                map(lambda x: x[2], nbrs[:self.max_neighbors]))
            nbr_feature[:len(nbrs)] = list(
                map(lambda x: x[1], nbrs[:self.max_neighbors]))
            neighbor_indexes.append(nbr_feature_idx)
            neighbor_features.append(nbr_feature)

        neighbor_indexes = numpy.array(neighbor_indexes)
        neighbor_features = numpy.array(neighbor_features)
        neighbor_features = self.gdf.expand2D(
            neighbor_features).astype(numpy.float32)

        return atom_feature, neighbor_features, neighbor_indexes
