from chainer_chemistry.dataset.preprocessors.common import construct_supernode_feature  # NOQA
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.gin_preprocessor import GINPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.rsgcn_preprocessor import RSGCNPreprocessor  # NOQA


class NFPGWMPreprocessor(NFPPreprocessor):
    def get_input_features(self, mol):
        atom_array, adj_array = super(
            NFPGWMPreprocessor, self).get_input_features(mol)
        super_node_x = construct_supernode_feature(
            mol, atom_array, adj_array)
        return atom_array, adj_array, super_node_x


class GGNNGWMPreprocessor(GGNNPreprocessor):
    def get_input_features(self, mol):
        atom_array, adj_array = super(
            GGNNGWMPreprocessor, self).get_input_features(mol)
        super_node_x = construct_supernode_feature(
            mol, atom_array, adj_array)
        return atom_array, adj_array, super_node_x


class GINGWMPreprocessor(GINPreprocessor):
    def get_input_features(self, mol):
        atom_array, adj_array = super(
            GINGWMPreprocessor, self).get_input_features(mol)
        super_node_x = construct_supernode_feature(
            mol, atom_array, adj_array)
        return atom_array, adj_array, super_node_x


class RSGCNGWMPreprocessor(RSGCNPreprocessor):
    def get_input_features(self, mol):
        atom_array, adj_array = super(
            RSGCNGWMPreprocessor, self).get_input_features(mol)
        super_node_x = construct_supernode_feature(
            mol, atom_array, adj_array)
        return atom_array, adj_array, super_node_x
