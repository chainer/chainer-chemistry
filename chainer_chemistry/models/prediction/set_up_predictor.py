from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA

import chainer  # NOQA

from chainer_chemistry.models.cgcnn import CGCNN
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.ggnn import GGNN
from chainer_chemistry.models.gin import GIN, GINSparse  # NOQA
from chainer_chemistry.models.gnn_film import GNNFiLM
from chainer_chemistry.models.megnet import MEGNet
from chainer_chemistry.models.mlp import MLP
from chainer_chemistry.models.nfp import NFP
from chainer_chemistry.models.prediction.graph_conv_predictor import GraphConvPredictor  # NOQA
from chainer_chemistry.models.relgat import RelGAT
from chainer_chemistry.models.relgcn import RelGCN, RelGCNSparse  # NOQA
from chainer_chemistry.models.rsgcn import RSGCN
from chainer_chemistry.models.schnet import SchNet
from chainer_chemistry.models.weavenet import WeaveNet


from chainer_chemistry.models.gwm.gwm_net import GGNN_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import GIN_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import NFP_GWM  # NOQA
from chainer_chemistry.models.gwm.gwm_net import RSGCN_GWM  # NOQA

from chainer_chemistry.models.cwle.cwle_net import GGNN_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import RelGAT_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import RelGCN_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import GIN_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import NFP_CWLE  # NOQA
from chainer_chemistry.models.cwle.cwle_net import RSGCN_CWLE  # NOQA

from chainer_chemistry.models.gwle.gwle_net import GGNN_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import RelGAT_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import RelGCN_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import GIN_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import NFP_GWLE  # NOQA
from chainer_chemistry.models.gwle.gwle_net import RSGCN_GWLE  # NOQA

from chainer_chemistry.models.cwle.cwle_graph_conv_model import MAX_WLE_NUM


def set_up_predictor(
        method,  # type: str
        n_unit,  # type: int
        conv_layers,  # type: int
        class_num,  # type: int
        label_scaler=None,  # type: Optional[chainer.Link]
        postprocess_fn=None,  # type: Optional[chainer.FunctionNode]
        n_atom_types=MAX_ATOMIC_NUM,
        conv_kwargs=None,  # type: Optional[Dict[str, Any]]
        n_wle_types=MAX_WLE_NUM  # type: int
):
    # type: (...) -> GraphConvPredictor
    """Set up the predictor, consisting of a GCN and a MLP.

    Args:
        method (str): Method name.
        n_unit (int): Number of hidden units.
        conv_layers (int): Number of convolutional layers for the graph
            convolution network.
        class_num (int): Number of output classes.
        label_scaler (chainer.Link or None): scaler link
        postprocess_fn (chainer.FunctionNode or None):
            postprocess function for prediction.
        conv_kwargs (dict): keyword args for GraphConvolution model.
    """
    mlp = MLP(out_dim=class_num, hidden_dim=n_unit)  # type: Optional[MLP]
    if conv_kwargs is None:
        conv_kwargs = {}

    if method == 'nfp' or method == 'nfp_wle':
        print('Set up NFP predictor...')
        conv = NFP(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'ggnn' or method == 'ggnn_wle':
        print('Set up GGNN predictor...')
        conv = GGNN(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'schnet':
        print('Set up SchNet predictor...')
        conv = SchNet(
            out_dim=class_num,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
        mlp = None
    elif method == 'weavenet':
        print('Set up WeaveNet predictor...')
        conv = WeaveNet(hidden_dim=n_unit, n_atom_types=n_atom_types, **conv_kwargs)
    elif method == 'rsgcn' or method == 'rsgcn_wle':
        print('Set up RSGCN predictor...')
        conv = RSGCN(out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'relgcn' or method == 'relgcn_wle':
        print('Set up Relational GCN predictor...')
        num_edge_type = 4
        conv = RelGCN(
            out_dim=n_unit,
            n_edge_types=num_edge_type,
            scale_adj=True,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'relgat' or method == 'relgat_wle':
        print('Set up Relational GAT predictor...')
        conv = RelGAT(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'gin' or method == 'gin_wle':
        print('Set up GIN predictor...')
        conv = GIN(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'nfp_gwm':
        print('Set up NFP_GWM predictor...')
        conv = NFP_GWM(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'ggnn_gwm':
        print('Set up GGNN_GWM predictor...')
        conv = GGNN_GWM(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'rsgcn_gwm':
        print('Set up RSGCN_GWM predictor...')
        conv = RSGCN_GWM(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'gin_gwm':
        print('Set up GIN_GWM predictor...')
        conv = GIN_GWM(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'nfp_cwle':
        print('Set up NFP_CWLE predictor...')
        conv = NFP_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'ggnn_cwle':
        print('Set up GGNN_CWLE predictor...')
        conv = GGNN_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'relgat_cwle':
        print('Set up RelGAT_CWLE predictor...')
        conv = RelGAT_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'relgcn_cwle':
        print('Set up RelGCN_CWLE predictor...')
        conv = RelGCN_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'rsgcn_cwle':
        print('Set up RSGCN_CWLE predictor...')
        conv = RSGCN_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'gin_cwle':
        print('Set up GIN_CWLE predictor...')
        conv = GIN_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'nfp_gwle':
        print('Set up NFP_GWLE predictor...')
        conv = NFP_GWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'ggnn_gwle':
        print('Set up GGNN_GWLE predictor...')
        conv = GGNN_GWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'relgat_gwle':
        print('Set up RelGAT_GWLE predictor...')
        conv = RelGAT_GWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'relgcn_gwle':
        print('Set up RelGCN_GWLE predictor...')
        conv = RelGCN_GWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'rsgcn_gwle':
        print('Set up RSGCN_GWLE predictor...')
        conv = RSGCN_GWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'gin_cwle':
        print('Set up GIN_CWLE predictor...')
        conv = GIN_CWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'gin_gwle':
        print('Set up GIN_gWLE predictor...')
        conv = GIN_GWLE(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            n_wle_types=n_wle_types,
            **conv_kwargs)
    elif method == 'relgcn_sparse':
        print('Set up RelGCNSparse predictor...')
        conv = RelGCNSparse(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'gin_sparse':
        print('Set up GIN predictor...')
        conv = GINSparse(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'gnnfilm':
        print('Training a GNN_FiLM predictor...')
        conv = GNNFiLM(
            out_dim=n_unit,
            hidden_channels=n_unit,
            n_update_layers=conv_layers,
            n_edge_types=5,
            n_atom_types=n_atom_types,
            **conv_kwargs)
    elif method == 'megnet':
        print('Set up MEGNet predictor...')
        conv = MEGNet(
            out_dim=n_unit,
            n_update_layers=conv_layers,
            **conv_kwargs)
    elif method == 'cgcnn':
        print('Set up CGCNN predictor...')
        conv = CGCNN(
            out_dim=n_unit,
            n_update_layers=conv_layers,
            **conv_kwargs)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))

    predictor = GraphConvPredictor(conv, mlp, label_scaler, postprocess_fn)
    return predictor
