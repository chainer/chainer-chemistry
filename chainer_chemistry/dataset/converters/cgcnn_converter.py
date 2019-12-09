import numpy

import chainer
from chainer.dataset.convert import to_device
from chainer import functions


@chainer.dataset.converter()
def cgcnn_converter(batch, device=None, padding=None):
    """CGCNN converter"""
    if len(batch) == 0:
        raise ValueError("batch is empty")

    atom_feat, nbr_feat, nbr_idx = [], [], []
    batch_atom_idx, target = [], []
    current_idx = 0
    xp = device.xp
    for element in batch:
        atom_feat.append(element[0])
        nbr_feat.append(element[1])
        nbr_idx.append(element[2] + current_idx)
        target.append(element[3])
        n_atom = element[0].shape[0]
        atom_idx = numpy.arange(n_atom) + current_idx
        batch_atom_idx.append(atom_idx)
        current_idx += n_atom

    atom_feat = to_device(device, functions.concat(atom_feat, axis=0).data)
    nbr_feat = to_device(device, functions.concat(nbr_feat, axis=0).data)
    # Always use numpy array for batch_atom_index
    # this is list of variable length array
    batch_atom_idx = numpy.array(batch_atom_idx)
    nbr_idx = to_device(device, functions.concat(nbr_idx, axis=0).data)
    target = to_device(device, xp.asarray(target))
    result = (atom_feat, nbr_feat, batch_atom_idx, nbr_idx, target)
    return result
