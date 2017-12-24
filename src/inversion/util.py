"""Utility functions for compatibility.

These functions mirror :mod:`numpy` functions but produce dask output.
"""
import itertools
import numbers
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lgmres
from scipy.sparse.linalg.interface import (
    MatrixLinearOperator,
    _CustomLinearOperator, _SumLinearOperator,
    _ScaledLinearOperator)
import dask.array as da
import dask.array.linalg as la
from dask.array import asarray, concatenate, stack, hstack

OPTIMAL_ELEMENTS = int(2e4)
"""Optimal elements per chunk in a dask array.

Magic number, arbitrarily chosen.  Dask documentation mentions many
chunks should fit easily in memory, but each should contain at least a
million elements.  This size matrix is fast to allocate and fill, but
:math:`10^5` gives a memory error.
"""
ARRAY_TYPES = (np.ndarray, da.Array)
"""Array types for determining Kronecker product type.

These are combined for a direct product.
"""
REAL_DTYPE_KINDS = "fiu"


def chunk_sizes(shape, matrix_side=True):
    """Good chunk sizes for the given shape.

    Optimized mostly for matrix operations on covariance matrices on
    this domain.

    Parameters
    ----------
    shape: tuple
    matrix_side: bool
        Whether the shape will need to be one side of a matrix or
        intended to stay a vector.

    Returns
    -------
    chunks: tuple
    """
    chunk_start = len(shape)
    nelements = 1

    here_max = OPTIMAL_ELEMENTS
    if not matrix_side:
        here_max **= 2

    if np.prod(shape) <= here_max:
        # The total number of elements is smaller than the recommended
        # chunksize, so the whole thing is a single chunk
        return tuple(shape)

    for dim_size in reversed(shape):
        nelements *= dim_size
        chunk_start -= 1

        if nelements > here_max:
            nelements //= dim_size
            break

    chunks = [1 if i < chunk_start else dim_size
              for i, dim_size in enumerate(shape)]
    next_dimsize = shape[chunk_start]

    # I happen to like neatness
    # And cholesky requires square chunks.
    if here_max > nelements:
        max_to_check = int(here_max // nelements)
        check_step = int(min(10 ** math.floor(math.log10(max_to_check) - .1),
                             here_max))
        chunks[chunk_start] = max(
            i for i in itertools.chain(
                (1,),
                range(check_step, max_to_check + 1, check_step))
            if next_dimsize % i == 0)
    return tuple(chunks)


def atleast_1d(arry):
    """Ensure `arry` is dask array of rank at least one.

    Parameters
    ----------
    arry: array_like

    Returns
    -------
    new_arry: dask.array.core.Array
    """
    if isinstance(arry, da.Array):
        if arry.ndim >= 1:
            return arry
        return arry[np.newaxis]
    if isinstance(arry, (list, tuple, np.ndarray)):
        arry = np.atleast_1d(arry)
    if isinstance(arry, numbers.Number):
        arry = np.atleast_1d(arry)

    array_shape = arry.shape
    return da.from_array(arry, chunks=chunk_sizes(array_shape))


def atleast_2d(arry):
    """Ensure arry is a dask array of rank at least two.

    Parameters
    ----------
    arry: array_like

    Returns
    -------
    new_arry: dask.array.core.Array
    """
    if isinstance(arry, da.Array):
        if arry.ndim >= 2:
            return arry
        elif arry.ndim == 1:
            return arry[np.newaxis, :]
        return arry[np.newaxis, np.newaxis]
    if isinstance(arry, (list, tuple, np.ndarray)):
        arry = np.atleast_2d(arry)
    if isinstance(arry, numbers.Number):
        arry = np.atleast_2d(arry)

    array_shape = arry.shape
    return da.from_array(arry, chunks=chunk_sizes(array_shape))


def linop_solve(operator, arr):
    """Solve `operator @ x = arr`.

    deal with arr possibly having multiple columns.

    Parameters
    ----------
    operator: LinearOperator
    arr: array_like

    Returns
    -------
    array_like
    """
    if arr.ndim == 1:
        return lgmres(operator, arr)[0]
    return stack([lgmres(operator, col)[0]
                  for col in atleast_2d(arr).T],
                 axis=1)


def solve(arr1, arr2):
    """Solve arr1 @ x = arr2 for x.

    Parameters
    ----------
    arr1: array_like[N, N]
    arr2: array_like[N]

    Returns
    -------
    array_like[N]
    """
    if hasattr(arr1, "solve"):
        return arr1.solve(arr2)
    elif isinstance(arr1, (MatrixLinearOperator, DaskMatrixLinearOperator)):
        return la.solve(asarray(arr1.A), asarray(arr2))
    elif isinstance(arr1, LinearOperator):
        # Linear operators with neither an underlying matrix nor a
        # provided solver. Use iterative sparse solvers.
        # TODO: Get preconditioner working
        # TODO: Test Ax = b for b not column vector
        if isinstance(arr2, ARRAY_TYPES):
            return linop_solve(arr1, arr2)
        elif isinstance(arr2, (MatrixLinearOperator,
                               DaskMatrixLinearOperator)):
            return linop_solve(arr1, arr2.A)
        else:
            def solver(vec):
                """Solve `arr1 x = vec`.

                Parameters
                ----------
                vec: array_like

                Returns
                -------
                array_like
                """
                return linop_solve(arr1, vec)
            inverse = DaskLinearOperator(matvec=solver, shape=arr1.shape[::-1])
            return inverse.dot(arr2)
        # # TODO: Figure out dask tasks for this
        # return da.Array(
        #     {(chunkname, 0):
        #      (spsolve, arr1, arr2.rechunk(1, whatever))},
        #     "solve-arr1.name-arr2.name",
        #     chunks)
    # Shorter method for assuring dask arrays
    return la.solve(asarray(arr1), asarray(arr2))


# TODO Test for handling of different chunking schemes
def schmidt_decomposition(vector, dim1, dim2):
    """Decompose a state vector into a sum of Kronecker products.

    Parameters
    ----------
    vector: array_like[dim1 * dim2]
    dim1, dim2: int

    Returns
    -------
    tuple of (weights, unit_vecs[dim1], unit_vecs[dim2]
        The rows form the separate vectors.
        The weights are guaranteed to be greater than zero

    Note
    ----
    Algorithm from stackexchange:
    https://physics.stackexchange.com/questions/251522/how-do-you-find-a-schmidt-basis-and-how-can-the-schmidt-decomposition-be-used-f
    Also from Mathematica code I wrote based on description in the green
    Quantum Computation book in the reading library
    """
    if vector.ndim == 2 and vector.shape[1] != 1:
        raise ValueError("Schmidt decomposition only valid for vectors")
    state_matrix = asarray(vector).reshape(dim1, dim2)

    if dim1 > dim2:
        vecs1, lambdas, vecs2 = la.svd(state_matrix)
    else:
        # Transpose, because dask expects tall and skinny
        vecs2T, lambdas, vecs1T = la.svd(state_matrix.T)
        vecs1 = vecs1T.T
        vecs2 = vecs2T.T

    return lambdas, vecs1.T[:len(lambdas), :], vecs2[:len(lambdas), :]


def kronecker_product(operator1, operator2):
    """Form the Kronecker product of the given operators.

    Delegates to ``operator1.kron()`` if possible,
    :func:`kron` if both are :const:`ARRAY_TYPES`, or
    :class:`inversion.correlations.KroneckerProduct` otherwise.

    Parameters
    ----------
    operator1, operator2: scipy.sparse.linalg.LinearOperator
        The component operators of the Kronecker product.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
    """
    if hasattr(operator1, "kron"):
        return operator1.kron(operator2)
    elif (isinstance(operator1, ARRAY_TYPES) and
          isinstance(operator2, ARRAY_TYPES)):
        return kron(operator1, operator2)
    from inversion.correlations import KroneckerProduct
    return KroneckerProduct(operator1, operator2)


def is_odd(num):
    """Return oddity of num.

    Parameters
    ----------
    num: int

    Returns
    -------
    bool
    """
    return num & 1 == 1


def tolinearoperator(operator):
    """Return operator as a LinearOperator.

    Parameters
    ----------
    operator: array_like or scipy.sparse.linalg.LinearOperator

    Returns
    -------
    DaskLinearOperator
        I want everything to work with dask where possible.

    See Also
    --------
    scipy.sparse.linalg.aslinearoperator
        A similar function without as wide a range of inputs.
        Used for everything but array_likes that are not
        :class:`np.ndarrays` or :class:`scipy.sparse.spmatrix`.
    """
    if isinstance(operator, ARRAY_TYPES):
        return DaskMatrixLinearOperator(atleast_2d(operator))
    try:
        return DaskLinearOperator.fromlinearoperator(
            aslinearoperator(operator))
    except TypeError:
        return DaskMatrixLinearOperator(atleast_2d(operator))


def kron(matrix1, matrix2):
    """Kronecker product of two matrices.

    Parameters
    ----------
    matrix1: array_like[M, N]
    matrix2: array_like[J, K]

    Returns
    -------
    array_like[M*J, N*K]

    See Also
    --------
    scipy.linalg.kron
        Where I got the overview of the implementation.
    """
    matrix1 = atleast_2d(matrix1)
    matrix2 = atleast_2d(matrix2)

    total_shape = matrix1.shape + matrix2.shape
    change = matrix1.ndim

    matrix1_index = tuple(slice(None) if i < change else np.newaxis
                          for i in range(len(total_shape)))
    matrix2_index = tuple(np.newaxis if i < change else slice(None)
                          for i in range(len(total_shape)))

    # TODO: choose good chunk sizes
    product = matrix1[matrix1_index] * matrix2[matrix2_index]

    return concatenate(concatenate(product, axis=1), axis=1)


class DaskLinearOperator(LinearOperator):
    """LinearOperator designed to work with dask.

    Does not support :class:`np.matrix` objects.
    """

    @classmethod
    def fromlinearoperator(cls, original):
        """Turn other into a DaskLinearOperator.

        Parameters
        ----------
        original: LinearOperator

        Returns
        -------
        DaskLinearOperator
        """
        if isinstance(original, DaskLinearOperator):
            return original
        # This returns _CustomLinearOperator, not DaskLinearOperator
        result = cls(
            matvec=original._matvec, rmatvec=original._rmatvec,
            matmat=original._matmat,
            shape=original.shape, dtype=original.dtype)
        return result

    # Everything below here essentially copied from the original LinearOperator
    # scipy.sparse.linalg.interface
    def __new__(cls, *args, **kwargs):
        if cls is DaskLinearOperator:
            # Operate as _DaskCustomLinearOperator factory.
            return _DaskCustomLinearOperator(*args, **kwargs)
        else:
            obj = super(DaskLinearOperator, cls).__new__(cls)

            if ((type(obj)._matvec == DaskLinearOperator._matvec and
                 type(obj)._matmat == DaskLinearOperator._matmat)):
                raise TypeError("LinearOperator subclass should implement"
                                " at least one of _matvec and _matmat.")

            return obj

    def _matmat(self, X):
        """Multiply self by matrix X using basic algorithm.

        Default matrix-matrix multiplication handler.

        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """
        return hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])

    def matvec(self, x):
        """

        Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : da.Array
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : da.Array
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """
        x = asarray(x)

        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        y = asarray(y)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : da.Array
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : da.Array
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """
        x = asarray(x)

        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        y = asarray(y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError(
                'invalid shape returned by user-defined rmatvec()')

        return y

    def matmat(self, X):
        """Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : da.Array
            An array with shape (N,K).

        Returns
        -------
        Y : da.Array
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """
        X = asarray(X)

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        M, N = self.shape

        if X.shape[0] != N:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._matmat(X)

        if isinstance(Y, np.matrix):
            Y = np.asmatrix(Y)

        return Y

    # Dask assumes everything has an ndim
    # Hopefully this fixes the problem
    ndim = 2

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _DaskSumLinearOperator(
                self, DaskLinearOperator.fromlinearoperator(x))
        elif isinstance(x, ARRAY_TYPES):
            return _DaskSumLinearOperator(
                self, DaskMatrixLinearOperator(x))
        else:
            return NotImplemented

    __radd__ = __add__

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.

        """
        if isinstance(x, (LinearOperator, DaskLinearOperator)):
            return ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _DaskScaledLinearOperator(self, x)
        else:
            x = asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def _adjoint(self):
        """Default implementation of _adjoint; defers to rmatvec."""
        shape = (self.shape[1], self.shape[0])
        return _DaskCustomLinearOperator(shape, matvec=self.rmatvec,
                                         rmatvec=self.matvec,
                                         dtype=self.dtype)


class _DaskCustomLinearOperator(DaskLinearOperator, _CustomLinearOperator):
    """This should let the factory functions above work."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=None):
        super(_DaskCustomLinearOperator, self).__init__(
            shape, matvec, rmatvec, matmat, dtype)

        if ((self.dtype.kind in REAL_DTYPE_KINDS and
             not hasattr(self, "_transpose"))):
            self._transpose = self._adjoint


class DaskMatrixLinearOperator(MatrixLinearOperator, DaskLinearOperator):
    """This should help out with the tolinearoperator.

    Should I override __add__, ...?
    """

    def __init__(self, A):
        super(DaskMatrixLinearOperator, self).__init__(A)
        self.__transp = None
        self.__adj = None

    def _transpose(self):
        if self.__transp is None:
            self.__transp = _DaskTransposeLinearOperator(self)
        return self.__transp

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _DaskAdjointLinearOperator(self)
        return self.__adj


class _DaskTransposeLinearOperator(DaskMatrixLinearOperator):
    """Transpose of a DaskMatrixLinearOperator."""

    def __init__(self, transpose):
        super(_DaskTransposeLinearOperator, self).__init__(transpose.A.T)
        self.__transp = transpose

    def _transpose(self):
        return self.__transp


class _DaskAdjointLinearOperator(DaskMatrixLinearOperator):
    """Adjoint of a DaskMatrixLinearOperator."""

    def __init__(self, adjoint):
        super(_DaskAdjointLinearOperator, self).__init__(adjoint.A.H)
        self.__adjoint = adjoint

    def _adjoint(self):
        return self.__adjoint


class _DaskSumLinearOperator(_SumLinearOperator, DaskLinearOperator):
    """Sum of two DaskLinearOperators."""

    pass


class ProductLinearOperator(LinearOperator):
    """Represent a product of linear operators."""

    def __init__(self, *operators):
        """Set up a product on linear operators.

        Parameters
        ----------
        operators: LinearOperator
        """
        super(ProductLinearOperator, self).__init__(
            None, (operators[0].shape[0], operators[-1].shape[1]))
        self._operators = tuple(tolinearoperator(op)
                                for op in operators)
        self._init_dtype()

    def _matvec(self, vector):
        """The matrix-vector product with vector.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        for op in reversed(self._operators):
            vector = op.matvec(vector)

        return vector

    def _rmatvec(self, vector):
        """Matrix-vector product on the left.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        for op in self._operators:
            vector = op.H.matvec(vector)

        return vector

    def _matmat(self, matrix):
        """The matrix-matrix product.

        Parameters
        ----------
        matrix: array_like

        Returns
        -------
        array_like
        """
        for op in reversed(self._operators):
            matrix = op.matmat(matrix)

        return matrix

    def _adjoint(self):
        """The Hermitian adjoint of the operator."""
        return ProductLinearOperator(
            [op.H for op in reversed(self._operators)])

    def _transpose(self):
        """The transpose of the operator."""
        return ProductLinearOperator(
            *[op.T for op in reversed(self._operators)])

    def solve(self, vector):
        """Solve A @ x == vector.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
            Solution of self @ x == vec
        """
        for op in self._operators:
            vector = solve(op, vector)

        return vector


class _DaskScaledLinearOperator(_ScaledLinearOperator, DaskLinearOperator):
    """Scaled linear operator."""

    pass
