import numpy
import pytest

from chainer_chemistry.utils.sparse_utils import convert_sparse_with_edge_type


num_edge_type = 4


def naive_convert(data, row, col, edge_type, num_edge_type):
    mb, length = data.shape
    new_mb = mb * num_edge_type
    new_data = [[] for _ in range(new_mb)]
    new_row = [[] for _ in range(new_mb)]
    new_col = [[] for _ in range(new_mb)]

    for i in range(mb):
        for j in range(length):
            k = i * num_edge_type + edge_type[i, j]
            new_data[k].append(data[i, j])
            new_row[k].append(row[i, j])
            new_col[k].append(col[i, j])

    new_length = max(len(arr) for arr in new_data)

    def pad(arr_2d, dtype=numpy.int32):
        for arr in arr_2d:
            arr.extend([0] * (new_length - len(arr)))
        return numpy.array(arr_2d)

    ret = []
    for d, r, c in zip(pad(new_data, data.dtype),
                       pad(new_row), pad(new_col)):
        ret.append(list(sorted(zip(d, r, c))))
    return ret


@pytest.mark.parametrize('in_shape,num_edge_type', [
    ((2, 4), 4),
    ((5, 10), 2),
    ((1, 1), 1),
    ((10, 1), 10),
    ((10, 10), 10),
])
def test_convert_sparse_with_edge_type(in_shape, num_edge_type):
    num_nodes = 10

    data = numpy.random.uniform(size=in_shape).astype(numpy.float32)
    row = numpy.random.randint(size=in_shape, low=0, high=num_nodes)
    col = numpy.random.randint(size=in_shape, low=0, high=num_nodes)
    edge_type = numpy.random.randint(size=in_shape, low=0, high=num_edge_type)

    received = convert_sparse_with_edge_type(data, row, col, num_nodes,
                                             edge_type, num_edge_type)
    expected = naive_convert(data, row, col, edge_type, num_edge_type)

    # check by minibatch-wise
    for i, expected_batch in enumerate(expected):
        d = received.data.data[i, :].tolist()
        r = received.row[i, :].tolist()
        c = received.col[i, :].tolist()

        received_batch = list(sorted(zip(d, r, c)))

        assert expected_batch == received_batch


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
