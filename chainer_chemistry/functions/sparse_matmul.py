import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import numpy
import warnings

try:
    from scipy import sparse
except ImportError:
    warnings.warn("SciPy seems not available on your system. A CPU"
                  " implementation of sparse_matmul uses SciPy, so that you"
                  " cannot use sparse_matmul on CPU.")


class sparse_coo_matrix(object):

    def __init__(self, data, row, col, shape,
                 use_variable=False, is_flatten=False):
        if len(shape) == 2:
            if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
                raise ValueError('ndim mismatch.')
        elif len(shape) == 3:
            _ndim = 2
            if is_flatten:
                _ndim = 1
            if data.ndim != _ndim or row.ndim != _ndim or col.ndim != _ndim:
                raise ValueError('ndim mismatch.')
        else:
            raise ValueError('ndim of tensor must be two or three.')

        self.data = data
        if use_variable:
            self.data = chainer.Variable(self.data)
        self.row = row
        self.col = col
        self.shape = shape  # ([nb], row, col)
        self.use_variable = use_variable
        self.is_flatten = is_flatten

    def to_dense(self):
        xp = cuda.get_array_module(self.data)
        # TODO(anaruse): need to consider a case when 'is_flatten' is True
        if len(self.shape) == 2:
            x = xp.zeros(self.shape, dtype=self.data.dtype)
            nz_idx = xp.where(self.row >= 0)
            x[self.row[nz_idx], self.col[nz_idx]] = self.data[nz_idx]
        elif len(self.shape) == 3:
            x = xp.zeros(self.shape, dtype=self.data.dtype)
            b_idx, nz_idx = xp.where(self.row >= 0)
            x[b_idx, self.row[b_idx, nz_idx],
              self.col[b_idx, nz_idx]] = self.data[b_idx, nz_idx]
        else:
            raise ValueError('ndim of tensor must be two or three.')
        return x

    def flatten(self):
        xp = cuda.get_array_module(self.data)
        if len(self.shape) == 2:
            return self
        elif len(self.shape) == 3:
            self_data = self.data
            if isinstance(self_data, chainer.Variable):
                self_data = self_data.data
            b_idx, nz_idx = xp.where(self.row >= 0)
            ft_data = self_data[b_idx, nz_idx]
            ft_row = self.row[b_idx, nz_idx] + b_idx * self.shape[-2]
            ft_col = self.col[b_idx, nz_idx] + b_idx * self.shape[-1]
            return sparse_coo_matrix(ft_data, ft_row, ft_col, self.shape,
                                     self.use_variable, is_flatten=True)
        else:
            raise ValueError('ndim of tensor must be two or three.')


def sparse_dense2coo(x, ldnz=None, use_variable=False):
    xp = cuda.get_array_module(x)
    if x.ndim == 2:
        _row, _col = xp.where(x != 0)
        nnz = len(_row)
        if ldnz is None or ldnz < nnz:
            ldnz = nnz
        data = xp.zeros((ldnz), dtype=x.dtype)
        row = xp.full((ldnz), -1, dtype=xp.int32)
        col = xp.full((ldnz), -1, dtype=xp.int32)
        data[:nnz] = x[_row, _col]
        row[:nnz] = xp.array(_row).astype(xp.int32)
        col[:nnz] = xp.array(_col).astype(xp.int32)
        shape = x.shape
        return sparse_coo_matrix(data, row, col, shape, use_variable)
    elif x.ndim == 3:
        # first axis is batch axis
        nb = x.shape[0]
        if ldnz is None:
            ldnz = 0
        for i in range(nb):
            ldnz = max(ldnz, len(xp.where(x[i] != 0)[0]))
        data = xp.empty((nb, ldnz), dtype=x.dtype)
        row = xp.empty((nb, ldnz), dtype=xp.int32)
        col = xp.empty((nb, ldnz), dtype=xp.int32)
        for i in range(nb):
            coo = sparse_dense2coo(x[i], ldnz)
            data[i] = coo.data
            row[i] = coo.row
            col[i] = coo.col
        shape = x.shape
        return sparse_coo_matrix(data, row, col, shape, use_variable)
    else:
        raise ValueError('ndim of x must be 2 or 3.')


def _sparse_matmul(a_data, a_row, a_col, a_shape, b,
                   transa, transb, transc, dtype=None):
    if dtype is None:
        dtype = numpy.result_type(a_data.dtype, b.dtype)
    if transa:
        a_row, a_col = a_col, a_row
        if len(a_shape) == 2:
            a_shape = [a_shape[-1], a_shape[-2]]
        else:
            a_shape = [a_shape[0], a_shape[-1], a_shape[-2]]
    if transb:
        b = b.swapaxes(-1, -2)

    xp = cuda.get_array_module(a_data, b)
    if xp is numpy:
        c = _sparse_matmul_cpu(a_data, a_row, a_col, a_shape, b, dtype)
    else:
        c = _sparse_matmul_gpu(a_data, a_row, a_col, a_shape, b, dtype)

    if transc:
        c = c.swapaxes(-1, -2)
    return c


def _sparse_matmul_cpu(A_data, A_row, A_col, A_shape, B, dtype):
    # A_shape: ([nb,] _m, _k)
    # B.shape: ([nb,] _k, _n)
    # A_data/row/col.shape: ([nb,] ldnz)
    _m, _k = A_shape[-2:]
    _n = B.shape[-1]
    if B.ndim == 2:
        sp_A = sparse.coo_matrix((A_data, (A_row, A_col)), shape=(_m, _k))
        C = sp_A.dot(B).astype(dtype, copy=False)
    elif B.ndim == 3:
        nb = B.shape[0]
        if A_data.ndim == 1:
            # sparse matrix A is flattened.
            _shape = [nb * _m, nb * _k]
            sp_A = sparse.coo_matrix((A_data, (A_row, A_col)), shape=_shape)
            _B = B.reshape((nb * _k, _n))
            _C = sp_A.dot(_B).astype(dtype, copy=False)
            C = _C.reshape((nb, _m, _n))
        elif A_data.ndim == 2:
            C = numpy.empty((nb, _m, _n), dtype=dtype)
            for i in range(nb):
                nnz = len(numpy.where(A_row[i] >= 0)[0])
                sp_A = sparse.coo_matrix((A_data[i, :nnz],
                                          (A_row[i, :nnz], A_col[i, :nnz])),
                                         shape=(_m, _k))
                C[i] = sp_A.dot(B[i]).astype(dtype, copy=False)
        else:
            raise ValueError('A_data.ndim must be one or two.')
    else:
        raise ValueError('B.ndim must be two or three')

    return C


def _sparse_matmul_gpu(A_data, A_row, A_col, A_shape, B, dtype):
    cupy_dtype = dtype
    if cupy_dtype == numpy.float16:
        cupy_dtype = numpy.float32
        # fp32 is used in cupy kernel because fp16 atomicAdd is not supported

    # A_shape: ([nb,] _m, _k)
    # B.shape: ([nb,] _k, _n)
    # A_data/row/col.shape: ([nb,] ldnz)
    _m, _k = A_shape[-2:]
    _n = B.shape[-1]
    ldnz = A_data.shape[-1]
    if B.ndim == 2:
        nb = 1
        C = cuda.cupy.zeros((_m, _n), dtype=cupy_dtype)
    elif B.ndim == 3:
        nb = B.shape[0]
        C = cuda.cupy.zeros((nb, _m, _n), dtype=cupy_dtype)
    else:
        raise ValueError('ndim of B must be two or three.')

    if len(A_shape) == 3 and A_data.ndim == 1:
        # sparse matrix A is flattened
        nthreads = ldnz * _n
        _cupy_sparse_matmul_ft()(nb, _m, _n, _k, ldnz, A_data, A_row, A_col,
                                 B, C, size=nthreads)
    else:
        nthreads = nb * ldnz * _n
        _cupy_sparse_matmul()(nb, _m, _n, _k, ldnz, A_data, A_row, A_col,
                              B, C, size=nthreads)

    return C.astype(dtype, copy=False)


def _cupy_sparse_matmul_ft():
    return cuda.cupy.ElementwiseKernel(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A _A_data, raw T _A_row, raw T _A_col, \
         raw B _B, raw C _C',
        '',
        '''
        int i_n = (i % _n);
        int i_nz = (i / _n);
        if (i_nz >= nnz) {
            continue;
        }
        int i_A = i_nz;

        int i_k = _A_col[i_A];
        if (i_k < 0) {
            continue;
        }
        assert(i_k < (nb * _k));

        int i_m = _A_row[i_A];
        if (i_m < 0) {
            continue;
        }
        assert(i_m < (nb * _m));

        int i_B = i_n + _n * i_k;
        int i_C = i_n + _n * i_m;
        A val_A = _A_data[i_A];
        B val_B = _B[i_B];
        atomicAdd(&_C[i_C], (C)(val_B * val_A));
        ''',
        'sparse_matmul_ft')


def _cupy_sparse_matmul():
    return cuda.cupy.ElementwiseKernel(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A _A_data, raw T _A_row, raw T _A_col, \
         raw B _B, raw C _C',
        '',
        '''
        int i_n = (i % _n);
        int i_A = (i / _n);
        int i_b = (i / _n) / nnz;
        if (i_b >= nb) {
            continue;
        }
        int i_k = _A_col[i_A];
        if (i_k < 0) {
            continue;
        }
        assert(i_k < _k);
        int i_m = _A_row[i_A];
        if (i_m < 0) {
            continue;
        }
        assert(i_m < _m);
        int i_B = i_n + _n * (i_k + _k * i_b);
        int i_C = i_n + _n * (i_m + _m * i_b);
        A val_A = _A_data[i_A];
        atomicAdd(&_C[i_C], (C)(_B[i_B] * val_A));
        ''',
        'sparse_matmul')


class SparseMatMul(function_node.FunctionNode):

    def __init__(self, sp_row, sp_col, sp_shape,
                 transa=False, transb=False, transc=False,
                 dtype=None):
        if sp_row.ndim != sp_col.ndim:
            raise ValueError('ndim of sp_row and sp_col must be the same.')
        if sp_row.ndim != 1 and sp_row.ndim != 2:
            raise ValueError('ndim of sp_row and sp_col must be one or two.')
        for i in range(sp_row.ndim):
            if sp_row.shape[i] != sp_col.shape[i]:
                _msg = 'shape of sp_row and sp_col must be the same.'
                raise ValueError(_msg)
        if len(sp_shape) != 2 and len(sp_shape) != 3:
            raise ValueError('len(sp_shape) must be two or three.')
        self.sp_row = sp_row  # ([nb,] ldnz)
        self.sp_col = sp_col  # ([nb,] ldnz)
        self.sp_shape = sp_shape  # ([nb,] _m, _k) when transa is False
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        sp_type, dn_type = in_types
        # sp_type.shape: ([nb,] ldnz)
        # dn_type.shape: ([nb,] _k, _n) when transb is False
        sp_k_axis = -1
        if self.transa:
            sp_k_axis = -2
        dn_k_axis = -2
        if self.transb:
            dn_k_axis = -1
        type_check.expect(
            sp_type.dtype.kind == 'f',
            dn_type.dtype.kind == 'f',
            dn_type.ndim == len(self.sp_shape),
            sp_type.ndim == self.sp_row.ndim,
            sp_type.ndim < dn_type.ndim,
            sp_type.shape[-1] == self.sp_row.shape[-1],
            self.sp_shape[sp_k_axis] == dn_type.shape[dn_k_axis],
        )
        dn_ndim = type_check.eval(dn_type.ndim)
        if dn_ndim == 3:
            type_check.expect(
                dn_type.shape[0] == self.sp_shape[0],
            )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        sp, dn = inputs
        c = _sparse_matmul(sp, self.sp_row, self.sp_col, self.sp_shape, dn,
                           self.transa, self.transb, self.transc, self.dtype)
        return utils.force_array(c, self.dtype),

    def backward(self, indexes, grad_outputs):
        sp, dn = self.get_retained_inputs()
        g_c, = grad_outputs

        g_sp = None
        if 0 in indexes:
            g_sp = SparseMatMulGradSP(self.sp_row, self.sp_col, self.sp_shape,
                                      self.transc, not self.transb,
                                      self.transa,
                                      dtype=sp.dtype).apply((g_c, dn))[0]

        g_dn = None
        if 1 in indexes:
            g_dn = SparseMatMul(self.sp_row, self.sp_col, self.sp_shape,
                                not self.transa, self.transc, self.transb,
                                dtype=dn.dtype).apply((sp, g_c))[0]

        return g_sp, g_dn


def _sparse_matmul_gradsp(a, b, c_row, c_col, c_shape,
                          transa, transb, transc, dtype):
    if dtype is None:
        dtype = numpy.result_type(a.dtype, b.dtype)
    if transa and transb:
        a, b = b, a
        transa, transb = not transb, not transa
        transc = not transc
    if transa:
        a = a.swapaxes(-1, -2)
    if transb:
        b = b.swapaxes(-1, -2)
    if transc:
        c_row, c_col = c_col, c_row
        if len(c_shape) == 2:
            c_shape = [c_shape[-1], c_shape[-2]]
        else:
            c_shape = [c_shape[0], c_shape[-1], c_shape[-2]]

    xp = cuda.get_array_module(a, b)
    if xp is numpy:
        return _sparse_matmul_gradsp_cpu(a, b, c_row, c_col, c_shape, dtype)
    else:
        return _sparse_matmul_gradsp_gpu(a, b, c_row, c_col, c_shape, dtype)


def _sparse_matmul_gradsp_cpu(A, B, C_row, C_col, C_shape, dtype):
    # A.shape: ([nb,] _m, _k)
    # B.shape: ([nb,] _k, _n)
    # C_row/col.shape: ([nb,] ldnz)
    # C_shape: ([nb,] _m, _n)
    _m, _k = A.shape[-2:]
    _n = B.shape[-1]
    ldnz = C_row.shape[-1]
    C = numpy.matmul(A, B).astype(dtype, copy=False)
    if A.ndim == 2:
        C_data = numpy.zeros((ldnz), dtype=dtype)
        nnz = len(numpy.where(C_row >= 0)[0])
        C_data[:nnz] = C[C_row[:nnz], C_col[:nnz]]
    elif A.ndim == 3:
        nb = A.shape[0]
        if C_row.ndim == 1:
            # sparse matrix C is flattend
            b_idx = C_row // _m
            c_row = C_row % _m
            c_col = C_col % _n
            C_data = C[b_idx, c_row, c_col]
        elif C_row.ndim == 2:
            C_data = numpy.zeros((nb, ldnz), dtype=dtype)
            b_idx, nz_idx = numpy.where(C_row >= 0)
            C_data[b_idx, nz_idx] = C[b_idx, C_row[b_idx, nz_idx],
                                      C_col[b_idx, nz_idx]]
        else:
            raise ValueError('ndim of C_row/col must be two or three.')
    else:
        raise ValueError('ndim of A must be two or three.')

    return C_data


def _sparse_matmul_gradsp_gpu(A, B, C_row, C_col, C_shape, dtype):
    # A.shape: ([nb,] _m, _k)
    # B.shape: ([nb,] _k, _n)
    # C_row/col.shape: ([nb,] ldnz)
    # C_shape: ([nb,] _m, _n)
    _m, _k = A.shape[-2:]
    _n = B.shape[-1]
    ldnz = C_row.shape[-1]
    nb = 1
    if len(C_shape) == 3:
        nb = C_shape[0]

    C_data = cuda.cupy.zeros(C_row.shape, dtype=dtype)

    if len(C_shape) == 3 and C_row.ndim == 1:
        # sparse matrix C is flattened.
        nthreads = ldnz
        _cupy_sparse_matmul_gradsp_ft()(nb, _m, _n, _k, ldnz, A, B,
                                        C_data, C_row, C_col, size=nthreads)
    else:
        nthreads = nb * ldnz
        _cupy_sparse_matmul_gradsp()(nb, _m, _n, _k, ldnz, A, B,
                                     C_data, C_row, C_col, size=nthreads)

    return C_data


def _cupy_sparse_matmul_gradsp_ft():
    return cuda.cupy.ElementwiseKernel(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A _A, raw B _B, \
         raw C _C_data, raw T _C_row, raw T _C_col',
        '',
        '''
        int i_nz = i;
        if (i_nz >= nnz) {
            continue;
        }
        int i_C = i_nz;

        int i_m = _C_row[i_C];
        if (i_m < 0) {
            continue;
        }
        assert(i_m < (nb * _m));
        int i_b1 = i_m / _m;
        i_m = i_m % _m;

        int i_n = _C_col[i_C];
        if (i_n < 0) {
            continue;
        }
        assert(i_n < (nb * _n));
        int i_b2 = i_n / _n;
        i_n = i_n % _n;

        assert(i_b1 == i_b2);

        C val_C = 0.0;
        for (int i_k = 0; i_k < _k; i_k++) {
            int i_A = i_k + _k * (i_m + _m * i_b1);
            int i_B = i_n + _n * (i_k + _k * i_b1);
            A val_A = _A[i_A];
            B val_B = _B[i_B];
            val_C += (C)(val_A * val_B);
        }
        _C_data[i_C] = val_C;
        ''',
        'sparse_matmul_gradsp_ft')


def _cupy_sparse_matmul_gradsp():
    return cuda.cupy.ElementwiseKernel(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A _A, raw B _B, \
         raw C _C_data, raw T _C_row, raw T _C_col',
        '',
        '''
        int i_nz = (i % nnz);
        int i_b = (i / nnz);
        if (i_b >= nb) {
            continue;
        }
        int i_C = i;
        int i_m = _C_row[i_C];
        if (i_m < 0) {
            continue;
        }
        assert(i_m < _m);
        int i_n = _C_col[i_C];
        if (i_n < 0) {
            continue;
        }
        assert(i_n < _n);
        C val_C = 0.0;
        for (int i_k = 0; i_k < _k; i_k++) {
            int i_A = i_k + _k * (i_m + _m * i_b);
            int i_B = i_n + _n * (i_k + _k * i_b);
            A val_A = _A[i_A];
            B val_B = _B[i_B];
            val_C += (C)(val_A * val_B);
        }
        _C_data[i_C] = val_C;
        ''',
        'sparse_matmul_gradsp')


class SparseMatMulGradSP(function_node.FunctionNode):

    def __init__(self, sp_row, sp_col, sp_shape,
                 transa=False, transb=False, transc=False,
                 dtype=None):
        if sp_row.ndim != sp_col.ndim:
            raise ValueError('ndim of sp_row and sp_col must be the same.')
        if sp_row.ndim != 1 and sp_row.ndim != 2:
            raise ValueError('ndim of sp_row and sp_col must be one or two.')
        for i in range(sp_row.ndim):
            if sp_row.shape[i] != sp_col.shape[i]:
                _msg = 'shape of sp_row and sp_col must be the same.'
                raise ValueError(_msg)
        if len(sp_shape) != 2 and len(sp_shape) != 3:
            raise ValueError('len(sp_shape) must be two or three.')
        self.sp_row = sp_row  # ([nb,] ldnz)
        self.sp_col = sp_col  # ([nb,] ldnz)
        self.sp_shape = sp_shape  # ([nb,] _m, _n) when transc is False
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types
        # a_type.shape: ([nb,] _m, _k) when transa is False
        # b_type.shape: ([nb,] _k, _n) when transb is False
        a_m_axis, a_k_axis = -2, -1
        b_k_axis, b_n_axis = -2, -1
        sp_m_axis, sp_n_axis = -2, -1
        if self.transa:
            a_m_axis, a_k_axis = -1, -2
        if self.transb:
            b_k_axis, b_n_axis = -1, -2
        if self.transc:
            sp_m_axis, sp_n_axis = -1, -2
        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 2,
            a_type.ndim <= 3,
            a_type.ndim == b_type.ndim,
            a_type.shape[a_m_axis] == self.sp_shape[sp_m_axis],
            b_type.shape[b_n_axis] == self.sp_shape[sp_n_axis],
            a_type.shape[a_k_axis] == b_type.shape[b_k_axis],
        )
        a_ndim = type_check.eval(a_type.ndim)
        if a_ndim == 3:
            type_check.expect(
                a_type.shape[0] == self.sp_shape[0],
                b_type.shape[0] == self.sp_shape[0],
            )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        a, b = inputs
        c = _sparse_matmul_gradsp(a, b,
                                  self.sp_row, self.sp_col, self.sp_shape,
                                  self.transa, self.transb, self.transc,
                                  self.dtype)
        return utils.force_array(c),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        g_sp, = grad_outputs

        g_a = None
        if 0 in indexes:
            g_a = SparseMatMul(self.sp_row, self.sp_col, self.sp_shape,
                               self.transc, not self.transb, self.transa,
                               dtype=a.dtype).apply((g_sp, b))[0]

        g_b = None
        if 0 in indexes:
            g_b = SparseMatMul(self.sp_row, self.sp_col, self.sp_shape,
                               not self.transc, self.transa, not self.transb,
                               dtype=b.dtype).apply((g_sp, a))[0]

        return g_a, g_b


def sparse_matmul(a, b, transa=False, transb=False):
    """Computes the batched multiplication of sparse and dense matrix.

    The following use cases are supported:

        1. C (dense) = A (sparse) * B (dense)
        2. C (dense) = A (dense) * B (sparse)

    Args:
        a (Variable or sparse_coo_matrix): The left operand of mat-mul.
        b (Variable or sparse_coo_matrix): The right operand of mat-mul.
        transa (bool): If ``True``, each matrix in ``a`` will be transposed.
        transb (bool): If ``True``, each matrix in ``b`` will be transposed.

    Returns:
        _chainer.Variable: Result of batched mat-mul.
    """
    if (isinstance(a, sparse_coo_matrix) and
            isinstance(b, (chainer.Variable, numpy.ndarray, cuda.ndarray))):
        return SparseMatMul(a.row, a.col, a.shape,
                            transa=transa,
                            transb=transb,
                            transc=False).apply((a.data, b))[0]
    elif (isinstance(a, (chainer.Variable, numpy.ndarray, cuda.ndarray)) and
          isinstance(b, sparse_coo_matrix)):
        return SparseMatMul(b.row, b.col, b.shape,
                            transa=not transb,
                            transb=not transa,
                            transc=True).apply((b.data, a))[0]
    else:
        _msg = 'This combination of type of inputs is not supported.\n'
        _msg += '    a: {}\n'.format(type(a))
        _msg += '    b: {}\n'.format(type(b))
        raise ValueError(_msg)
