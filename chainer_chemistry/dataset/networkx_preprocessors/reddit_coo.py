import numpy
import scipy
import chainer
from chainer_chemistry.dataset.graph_dataset.base_graph_data import PaddingGraphData  # NOQA


def get_reddit_coo_data(dirpath):
    """Temporary function to obtain reddit coo data for GIN
    (because it takes to much time to convert it to networkx)

    Returns:
        PaddingGraphData: `PaddingGraphData` of reddit
    """

    print("Loading node feature and label")
    reddit_data = numpy.load(dirpath + "reddit_data.npz")

    print("Loading edge data")
    coo_adj = scipy.sparse.load_npz(dirpath + "reddit_graph.npz")
    row = coo_adj.row.astype(numpy.int32)
    col = coo_adj.col.astype(numpy.int32)
    data = coo_adj.data.astype(numpy.float32)

    # ensure row is sorted
    if not numpy.all(row[:-1] <= row[1:]):
        order = numpy.argsort(row)
        row = row[order]
        col = col[order]
    assert numpy.all(row[:-1] <= row[1:])

    adj = chainer.utils.CooMatrix(
        data=data, row=row, col=col,
        shape=coo_adj.shape,
        order='C')

    return PaddingGraphData(
        x=reddit_data['feature'].astype(numpy.float32),
        adj=adj,
        y=reddit_data['label'].astype(numpy.int32),
        label_num=41
    )
