import chainer
from chainer.dataset.convert import to_device


@chainer.dataset.converter()
def megnet_converter(batch, device=None, padding=0):
    """MEGNet converter"""
    if len(batch) == 0:
        raise ValueError("batch is empty")

    atom_feat, pair_feat, global_feat, target = [], [], [], []
    atom_idx, pair_idx, start_idx, end_idx = [], [], [], []
    batch_size = len(batch)
    current_atom_idx = 0
    for i in range(batch_size):
        element = batch[i]
        n_atom = element[0].shape[0]
        n_pair = element[1].shape[0]
        atom_feat.extend(element[0])
        pair_feat.extend(element[1])
        global_feat.append(element[2])
        atom_idx.extend([i]*n_atom)
        pair_idx.extend([i]*n_pair)
        start_idx.extend(element[3][0] + current_atom_idx)
        end_idx.extend(element[3][1] + current_atom_idx)
        target.append(element[4])
        current_atom_idx += n_atom

    xp = device.xp
    atom_feat = to_device(device, xp.asarray(atom_feat))
    pair_feat = to_device(device, xp.asarray(pair_feat))
    global_feat = to_device(device, xp.asarray(global_feat))
    atom_idx = to_device(device, xp.asarray(atom_idx))
    pair_idx = to_device(device, xp.asarray(pair_idx))
    start_idx = to_device(device, xp.asarray(start_idx))
    end_idx = to_device(device, xp.asarray(end_idx))
    target = to_device(device, xp.asarray(target))
    result = (atom_feat, pair_feat, global_feat, atom_idx, pair_idx,
              start_idx, end_idx, target)

    return result
