import os
import numpy
import json


from chainer_chemistry.dataset.converters.cgcnn_converter import cgcnn_converter  # NOQA
from chainer_chemistry.dataset.utils import GaussianDistance
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA


class CGCNNPreprocessor(MolPreprocessor):
    """CGCNNPreprocessor

    Args:
    For Molecule: TODO

    For Crystal
        data_dir : TODO (directory path which includes atom_init.json)
        max_num_nbr (int): Max number of atom considered as neighbors
        max_radius (float): Cutoff radius (angstrom)
        expand_dim (int): Dimension converting from distance to vector
    """

    def __init__(self, data_dir, max_num_nbr=12, max_radius=8, expand_dim=40):
        super(CGCNNPreprocessor, self).__init__()

        self.max_num_nbr = max_num_nbr
        self.max_radius = max_radius
        self.gdf = GaussianDistance(centers=numpy.linspace(0, 8, expand_dim))
        feat_dict = json.load(open(os.path.join(data_dir, "atom_init.json")))
        self.atom_features = {int(key): numpy.array(value,
                                                    dtype=numpy.float32)
                              for key, value in feat_dict.items()}

    def get_input_feature_from_crystal(self, structure):
        """get input features from structure object

        Args:
            structure (Structure):

        """

        # get atom feature vector
        atom_feature = numpy.vstack(
            [self.atom_features[structure[i].specie.number]
             for i in range(len(structure))]
        )

        # get edge feature vector & bond idx
        neighbor_indexes = []
        neighbor_features = []
        all_neighbors = structure.get_all_neighbors(self.max_radius,
                                                    include_index=True)
        all_neighbors = [sorted(nbrs, key=lambda x: x[1])
                         for nbrs in all_neighbors]

        for nbrs in all_neighbors:
            nbr_feature_idx = numpy.zeros(self.max_num_nbr, dtype=numpy.int32)
            nbr_feature = numpy.zeros(self.max_num_nbr, dtype=numpy.float32) \
                + self.max_radius + 1.
            nbr_feature_idx[:len(nbrs)] = [x[2]
                                           for x in nbrs[:self.max_num_nbr]]
            nbr_feature[:len(nbrs)] = [x[1] for x in nbrs[:self.max_num_nbr]]
            neighbor_indexes.append(nbr_feature_idx)
            neighbor_features.append(nbr_feature)

        neighbor_indexes = numpy.array(neighbor_indexes)
        neighbor_features = numpy.array(neighbor_features)
        neighbor_features = self.gdf.expand_from_distances(neighbor_features)

        return atom_feature, neighbor_features, neighbor_indexes

    def create_dataset(self, *args, **kwargs):
        # TODO (nakago): HACKING. override `converter` for cgcnn for now...
        dataset = super(CGCNNPreprocessor, self).create_dataset(
            *args, **kwargs)
        dataset.converter = cgcnn_converter
        return dataset
