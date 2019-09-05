import numpy as np


from chainer_chemistry.dataset.utils import GaussianDistance


MAX_ATOM_ELEMENT = 94


class MPMEGNetPreprocessor(object):
    """preprocessor class for MEGNet and crystal object

    """

    def __init__(self, max_neighbors=12, max_radius=8, exapand_dim=100):
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.rbf = GaussianDistance(centers=np.linspace(0, 5, exapand_dim))


    def get_input_feature(self, crystal):
        atom_num = len(crystal)
        atom_feature = np.zeros((atom_num, MAX_ATOM_ELEMENT), dtype=np.float32)
        for i in range(atom_num):
            if crystal[i].specie.number < MAX_ATOM_ELEMENT:
                atom_feature[i][crystal[i].specie.number] = 1

        # get edge feture vector & bond idx
        neighbor_indexes = []
        neighbor_features = []
        all_neighbors = crystal.get_all_neighbors(self.max_radius, include_index=True)
        all_neighbors = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbors]
        bond_num = len(all_neighbors)
        for i in range(bond_num):
            nbrs = all_neighbors[i]
            start_node_idx = i
            nbr_feature = np.zeros(self.max_neighbors, dtype=np.float32) + self.max_radius + 1.
            nbr_feature_idx = np.zeros((self.max_neighbors, 2), dtype=np.int32)
            nbr_feature_idx[:, 0] = start_node_idx
            nbr_feature_idx[:, 1] = list(map(lambda x: x[2], nbrs[:self.max_neighbors]))
            nbr_feature[:len(nbrs)] = list(map(lambda x: x[1], nbrs[:self.max_neighbors]))
            neighbor_indexes.append(nbr_feature_idx)
            neighbor_features.append(nbr_feature)

        bond_idx = np.array(neighbor_indexes).reshape(-1, 2).T
        neighbor_features = np.array(neighbor_features)
        # apply gaussian filter to neighbor distance
        neighbor_features = self.rbf.expand2D(neighbor_features).reshape(-1, 100)
        # get global feature vector
        global_feature = np.array([0, 0], dtype=np.float32)

        return atom_feature, neighbor_features, global_feature, atom_num, bond_num, bond_idx
