"""Correlations; used for assumed background covariances.

Multiply by diagonal matrices of the assumed standard deviations on
the right and left to obtain covariance matrices.

"""
import abc
import functools

# Need to specify numpy versions in some instances.
import numpy as np
# Not in dask.array
from numpy.linalg import eigh, norm
# arange changes signature
from numpy import arange, newaxis, asanyarray

from numpy import fromfunction, asarray, hstack, flip
from numpy import exp, square, fmin, sqrt
from numpy import logical_or, concatenate, isnan
from numpy import where
from numpy import sum as array_sum
from scipy.special import gamma, kv as K_nu
from scipy.sparse.linalg.interface import LinearOperator

import pyfftw.interfaces.cache
from pyfftw import next_fast_len
from pyfftw.interfaces.numpy_fft import rfftn, irfftn
import six

from .linalg import SelfAdjointLinearOperator
from .linalg import SchmidtKroneckerProduct
from .linalg_interface import is_odd

NUM_THREADS = 8
"""Number of threads :mod:`pyfftw` should use for each transform."""
ADVANCE_PLANNER_EFFORT = "FFTW_MEASURE"
"""Amount of advance planning FFTW should do.

This is done when the :class:`HomogeneousIsotropicCorrelation`
instance is created.  It is set to "FFTW_MEASURE" to keep test runs
fast, but applications should probably set this to do `a more
extensive exploration of the possibilities
<https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#configuring-fftw-planning-effort-and-number-of-threads>`_.
"""
PLANNER_EFFORT = "FFTW_ESTIMATE"
"""How much effort to expend planning for subsequent inversions.

Used by :meth:`HomogeneousIsotropicCorrelation.matmat`.  This is used
more frequently than :data:`ADVANCE_PLANNER_EFFORT` and should
probably stay set to "FFTW_ESTIMATE".  Increasing this can increase
application runtime by a factor of two.
"""
pyfftw.interfaces.cache.enable()

ROUNDOFF = 1e-13
"""Approximate size of roundoff error for correlation matrices.

Eigenvalues less than this value will be reset to this.

Gaussian correlations with a correlation length between five and ten
cells need `ROUNDOFF` greater than 1e-15 to be numerically positive
definite.

Gaussian(15) needs 1e-13 > ROUNDOFF > 1e-14

Also used in SchmidtKroneckerProduct to determine how many terms in the
Schmidt decomposition should be used.
"""
NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.
"""
FOURIER_NEAR_ZERO = 1e-15
"""Where fourier coefficients are treated as zero.

1e-20 produces overflow with the dask tests
"""
DTYPE = np.float32


class HomogeneousIsotropicCorrelation(SelfAdjointLinearOperator):
    """Homogeneous isotropic correlations using FFTs.

    This embeds the physical domain passed in a larger computational
    domain, which allows for treatment of domains that should not be
    considered periodic.

    .. note::

        Do not invoke this directly. Use
        :func:`HomogeneousIsotropicCorrelation.from_function` or
        :func:`HomogeneousIsotropicCorrelaiton.from_array` instead.

    See Also
    --------
    :func:`scipy.linalg.solve_circulant`
        I stole the idea from here.

    """

    def __init__(self, shape, computational_shape=None):
        """Set up the instance.

        .. note::

            Do not invoke this directly. Use
            :func:`HomogeneousIsotropicCorrelation.from_function` or
            :func:`HomogeneousIsotropicCorrelaiton.from_array` instead.

        Parameters
        ----------
        shape: tuple of int
            The state is formally input as a vector, but the correlations
            depend on the layout in some other shape, usually related to the
            physical layout. This is that shape.
        computational_shape: tuple of int
            The shape of the embedding computational domain.  Defaults
            to shape.  May be larger to induce non-periodic correlations
        """
        state_size = np.prod(shape)
        ndims = len(shape)

        super(HomogeneousIsotropicCorrelation, self).__init__(
            dtype=DTYPE, shape=(state_size, state_size))

        if computational_shape is None:
            computational_shape = shape

        is_cyclic = (shape == computational_shape)
        self._is_cyclic = is_cyclic
        self._underlying_shape = tuple(shape)
        self._computational_shape = computational_shape

        # Tell pylint these will be present
        self._corr_fourier = None
        self._fourier_near_zero = None

        self._fft = functools.partial(
            rfftn, axes=arange(0, ndims, dtype=int), s=computational_shape,
            threads=NUM_THREADS, planner_effort=PLANNER_EFFORT)

        if is_cyclic:
            self._ifft = functools.partial(
                irfftn, axes=arange(0, ndims, dtype=int), s=shape,
                threads=NUM_THREADS, planner_effort=PLANNER_EFFORT)
        else:
            axes = arange(0, ndims, dtype=int)
            base_slices = tuple(slice(None, dim) for dim in shape)

            def _ifft(arry):
                """Find inverse FFT of arry.

                Parameters
                ----------
                spectrum: array_like
                    The spectrum of a real-valued signal

                Returns
                -------
                array_like
                    The spectrum transformed back to physical space
                """
                slicer = (
                    base_slices +
                    tuple(slice(None) for dim in arry.shape[ndims:])
                )
                big_result = irfftn(
                    arry, axes=axes, s=computational_shape,
                    threads=NUM_THREADS, planner_effort=PLANNER_EFFORT)
                return big_result[slicer]

            self._ifft = _ifft

    @classmethod
    def from_function(cls, corr_func, shape, is_cyclic=True):
        """Create an instance to apply the correlation function.

        Parameters
        ----------
        corr_func: callable(dist) -> float
            The correlation of the first element of the domain with
            each other element.
        shape: tuple of int
            The state is formally a vector, but the correlations are
            assumed to depend on the layout in some other shape,
            usually related to the physical layout. This is the other
            shape.
        is_cyclic: bool
            Whether to assume the domain is periodic in all directions.

        Returns
        -------
        HomogeneousIsotropicCorrelation
        """
        shape = np.atleast_1d(shape)
        if is_cyclic:
            computational_shape = tuple(shape)
        else:
            computational_shape = tuple(next_fast_len(2 * dim - 1)
                                        for dim in shape)

        self = cls(tuple(shape), computational_shape)
        shape = np.asarray(self._computational_shape)
        ndims = len(shape)

        broadcastable_shape = shape[:, newaxis]
        while broadcastable_shape.ndim < ndims + 1:
            broadcastable_shape = broadcastable_shape[..., newaxis]

        def corr_from_index(*index):
            """Correlation of index with zero.

            Turns a correlation function in terms of index distance
            into one in terms of indices on a periodic domain.

            Parameters
            ----------
            index: tuple of int

            Returns
            -------
            float[-1, 1]
                The correlation of the given index with the origin.

            See Also
            --------
            DistanceCorrelationFunction.correlation_from_index
            """
            comp2_1 = square(index)
            # Components of distance to shifted origin
            comp2_2 = square(broadcastable_shape - index)
            # use the smaller components to get the distance to the
            # closest of the shifted origins
            comp2 = fmin(comp2_1, comp2_2)
            return corr_func(sqrt(array_sum(comp2, axis=0)))

        corr_struct = fromfunction(
            corr_from_index, shape=tuple(shape),
            dtype=DTYPE)

        # I should be able to generate this sequence with a type-I DCT
        # For some odd reason complex/complex is faster than complex/real
        # This also ensures the format here is the same as in _matmat
        corr_fourier = rfftn(
            corr_struct, axes=arange(ndims, dtype=int),
            threads=NUM_THREADS, planner_effort=ADVANCE_PLANNER_EFFORT)
        self._corr_fourier = (corr_fourier)

        # This is also affected by roundoff
        abs_corr_fourier = abs(corr_fourier)
        self._fourier_near_zero = (
            abs_corr_fourier < FOURIER_NEAR_ZERO * abs_corr_fourier.max())
        return self

    @classmethod
    def from_array(cls, corr_array, is_cyclic=True):
        """Create an instance with the given correlations.

        Parameters
        ----------
        corr_array: array_like
            The correlation of the first element of the domain with
            each other element.
        is_cyclic: bool

        Returns
        -------
        HomogeneousIsotropicCorrelation
        """
        corr_array = asarray(corr_array)
        shape = corr_array.shape
        ndims = corr_array.ndim

        if is_cyclic:
            computational_shape = shape
            self = cls(shape, computational_shape)
            corr_fourier = self._fft(corr_array)
        else:
            computational_shape = tuple(
                2 * (dim - 1) for dim in shape)
            self = cls(shape, computational_shape)

            for axis in reversed(range(ndims)):
                corr_array = concatenate(
                    [corr_array, flip(corr_array[1:-1], axis)],
                    axis=axis)

            # Advantages over dctn: guaranteed same format and gets
            # nice planning done for the later evaluations.
            corr_fourier = rfftn(
                corr_array, axes=arange(0, ndims, dtype=int),
                threads=NUM_THREADS, planner_effort=ADVANCE_PLANNER_EFFORT)

        # The fft axes need to be a single chunk for the dask ffts
        # It's in memory already anyway
        # TODO: create a from_spectrum to delegate to
        self._corr_fourier = (corr_fourier)
        self._fourier_near_zero = (corr_fourier < FOURIER_NEAR_ZERO)
        return self

    def sqrt(self):
        """Compute an S such that S.T @ S == self.

        Returns
        -------
        S: HomogeneousIsotropicCorrelation
        """
        result = HomogeneousIsotropicCorrelation(self._underlying_shape,
                                                 self._computational_shape)
        result._corr_fourier = sqrt(self._corr_fourier)
        # I still don't much trust these.
        result._fourier_near_zero = self._fourier_near_zero
        return result

    def _matmat(self, mat):
        """Evaluate the matrix product of self and `mat`.

        Parameters
        ----------
        mat: array_like[N, K]

        Returns
        -------
        array_like[N, K]
        """
        _shape = self._underlying_shape
        fields = mat.reshape(_shape + (-1,))

        if not isinstance(fields, np.ndarray):
            fields = fields.todense()

        spectral_fields = self._fft(fields)
        spectral_fields *= self._corr_fourier[..., np.newaxis]
        results = self._ifft(spectral_fields)

        return results.reshape(mat.shape)

    def inv(self):
        """Construct the matrix inverse of this operator.

        Returns
        -------
        LinearOperator

        Raises
        ------
        ValueError
            if the instance is acyclic.
        """
        if not self._is_cyclic:
            raise NotImplementedError(
                "HomogeneousIsotropicCorrelation.inv "
                "does not support acyclic correlations")
        # TODO: Return a HomogeneousIsotropicLinearOperator
        return LinearOperator(
            shape=self.shape, dtype=self.dtype,
            matvec=self.solve, rmatvec=self.solve)

    def solve(self, vec):
        """Solve A @ x = vec.

        Parameters
        ----------
        vec: array_like[N]

        Returns
        -------
        array_like[N]
            Solution of `self @ x = vec`

        Raises
        ------
        NotImplementedError
            if the instance is acyclic.
        ValueError
            If vec is the wrong shape
        """
        if not self._is_cyclic:
            raise NotImplementedError(
                "HomogeneousIsotropicCorrelation.solve "
                "does not support acyclic correlations")
        if vec.shape[0] != self.shape[0]:
            raise ValueError("Shape of vec not correct")
        field = asarray(vec).reshape(self._underlying_shape)

        spectral_field = self._fft(field)
        spectral_field /= self._corr_fourier
        # Dividing by a small number is numerically unstable. This is
        # nearly an SVD solve already, so borrow that solution.
        spectral_field[self._fourier_near_zero] = 0
        result = self._ifft(spectral_field)

        return result.reshape(self.shape[-1])

    def kron(self, other):
        """Construct the Kronecker product of this operator and other.

        Parameters
        ----------
        other: HomogeneousIsotropicCorrelation
            The other operator for the Kronecker product.
            This implementation will accept other objects,
            passing them along to :class:`.linalg.SchmidtKroneckerProduct`.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
        """
        if ((not isinstance(other, HomogeneousIsotropicCorrelation) or
             self._is_cyclic != other._is_cyclic)):
            return SchmidtKroneckerProduct(self, other)
        shape = self._underlying_shape + other._underlying_shape
        shift = len(self._underlying_shape)

        self_index = tuple(slice(None) if i < shift else np.newaxis
                           for i in range(len(shape)))
        other_index = tuple(np.newaxis if i < shift else slice(None)
                            for i in range(len(shape)))

        # rfft makes the first axis half the size
        # When combining things like this, I need to re-double that size again
        if is_odd(self._underlying_shape[-1]):
            reverse_start = -1
        else:
            reverse_start = -2
        expanded_fft = hstack(
            (self._corr_fourier,
             self._corr_fourier[..., reverse_start:0:-1].conj()))

        expanded_near_zero = concatenate(
            (self._fourier_near_zero,
             self._fourier_near_zero[..., reverse_start:0:-1]), axis=-1)

        newinst = HomogeneousIsotropicCorrelation(shape)
        newinst._corr_fourier = (expanded_fft[self_index] *
                                 other._corr_fourier[other_index])
        newinst._fourier_near_zero = logical_or(
            expanded_near_zero[self_index],
            other._fourier_near_zero[other_index])
        return newinst


def make_matrix(corr_func, shape):
    """Make a correlation matrix for a domain with shape `shape`.

    Parameters
    ----------
    corr_func: callable(float) -> float[-1, 1]
        Function giving correlation between two indices a distance d
        from each other.
    shape: tuple of int
        The underlying shape of the domain. It is viewed as a vector
        here, but may be more naturally seen as an N-D array. This is
        the shape of that array.
        `N = prod(shape)`

    See Also
    --------
    :func:`statsmodels.stats.correlation_tools.corr_clipped`
        Does something similar, and refers to other functions that may
        give more accurate results.

    Returns
    -------
    corr: np.ndarray[N, N]
        Positive definite dense array, entirely in memory
    """
    shape = tuple(np.atleast_1d(shape))
    n_points = np.prod(shape)

    # Since dask doesn't have eigh, using dask in this section slows
    # the test suite by about 25%.  Since it all ends up in memory,
    # may as well start with it there instead of converting back and
    # forth a few times.
    tmp_res = np.fromfunction(corr_func.correlation_from_index,
                              shape=2 * shape,
                              dtype=DTYPE).reshape(
        (n_points, n_points))
    where_small = tmp_res < NEAR_ZERO
    where_small &= tmp_res > -NEAR_ZERO

    # This isn't always positive definite.  I reset the values on
    # the negative side of roundoff to the positive side
    vals, vecs = eigh(tmp_res)
    del tmp_res
    vals[vals < ROUNDOFF] = ROUNDOFF

    result = np.dot(vecs, np.diag(vals).dot(vecs.T))

    # Now, there's more roundoff
    # make the values that were originally small zero
    result[where_small] = 0
    return asarray(result)


class DistanceCorrelationFunction(six.with_metaclass(abc.ABCMeta)):
    """A correlation function that depends only on distance."""

    _distance_scaling = 1

    def __init__(self, length):
        """Set up instance.

        Parameters
        ----------
        length: float
            The correlation length in index space. Unitless.
        """
        self._length = self._distance_scaling * float(length)

    def __repr__(self):
        """Return a string representation of self."""
        return "{name:s}({length:g})".format(
            length=self._length / self._distance_scaling,
            name=type(self).__name__)

    def __str__(self):
        """Return a string version for printing."""
        return "{name:s}({length:3.2e})".format(
            length=self._length / self._distance_scaling,
            name=type(self).__name__)

    @abc.abstractmethod
    def __call__(self, dist):
        """Get the correlation between points whose indices differ by dist.

        Parameters
        ----------
        dist: float
            The distance at which the correlation is requested.

        Returns
        -------
        correlation: float[-1, 1]
            The correlation at points that distance apart
        """
        pass  # pragma: no cover

    def correlation_from_index(self, *indices):
        """Find the correlation between the indices.

        Should be independent of the length of the underlying shape,
        but `indices` must still be even.

        Parameters
        ----------
        indices: tuple of int

        Returns
        -------
        float
            Find the correlation of the point contained in the first
            half of the indices with that contained in the second.
        """
        half = len(indices) // 2
        point1 = asanyarray(indices[:half])
        point2 = asanyarray(indices[half:])
        dist = norm(point1 - point2, axis=0)
        return self(dist)


class ExponentialCorrelation(DistanceCorrelationFunction):
    """A exponential correlation structure.

    Notes
    -----
    Correlation given by exp(-dist/length)
    where dist is the distance between the points.
    """

    def __call__(self, dist):
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        return exp(-dist / self._length)


class BalgovindCorrelation(DistanceCorrelationFunction):
    """A Balgovind 3D correlation structure.

    Follows Balgovind et al. 1983 recommendations for a 3D field,
    modified so the correlation length better matches that used by
    other correlation functions.

    Notes
    -----
    Correlation given by :math:`(1 + 2*dist/length) exp(-2*dist/length)`

    This implementation has problems for length == 10.
    I have no idea why.  3 and 30 are fine.

    References
    ----------
    Balgovind, R and Dalcher, A. and Ghil, M. and Kalnay, E. (1983).
    A Stochastic-Dynamic Model for the Spatial Structure of Forecast
    Error Statistics *Monthly Weather Review* 111(4) 701--722.
    :doi:`10.1175/1520-0493(1983)111<0701:asdmft>2.0.co;2`
    """

    _distance_scaling = 0.5

    def __call__(self, dist):
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float
            The distance between the points

        Returns
        -------
        float
            The correlation between points the given distance apart.
        """
        scaled_dist = dist / self._length
        return (1 + scaled_dist) * exp(-scaled_dist)


class MaternCorrelation(DistanceCorrelationFunction):
    r"""A Matern correlation structure.

    Follows Matern (1986) *Spatial Variation*.  The specific
    definition seems to be similar to the parameterization in Stern's
    *Interpolation of Spatial Data*

    Notes
    -----
    Correlation given by
    :math:`[2^{\kappa-1}\Gamma(\kappa)]^{-1} (d/L)^{\kappa} K_{\kappa}(d/L)`
    where :math:`\kappa` is a smoothness parameter and
    :math:`K_{\kappa}` is a modified Bessel function of the third kind.

    References
    ----------
    Stein, Michael L. *Interpolation of Spatial Data: Some Theory for
    Kridging* Springer-Verlag New York.  Part of Springer Series in
    Statistics (issn:0172-7397) isbn:978-1-4612-7166-6.
    :doi:`10.1007/978-1-4612-1494-6`
    """

    _distance_scaling = 1.25

    def __init__(self, length, kappa=1):
        r"""Set up instance.

        Parameters
        ----------
        length: float
            The correlation length in index space. Unitless.
        kappa: float
            The smoothness parameter
            :math:`kappa=\infty` is equivalent to Gaussian correlations
            :math:`kappa=\frac{1}{2}` is equivalent to exponential
            :math:`kappa=1` is Balgovind's recommendation for 2D fields
            :math:`kappa=\frac{3}{2}` matches Balgovind's advice for 3D fields
            Default value is only for full equivalence with other classes.
            The default value is entirely arbitrary and may change without
            notice.
        """
        self._kappa = kappa
        # Make sure correlation at zero is one
        self._scale_const = .5 * gamma(kappa)
        self._distance_scaling = self._distance_scaling * self._scale_const
        super(MaternCorrelation, self).__init__(length)

    def __call__(self, dist):
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        kappa = self._kappa
        scaled_dist = dist / self._length
        result = ((.5 * scaled_dist) ** kappa *
                  K_nu(kappa, scaled_dist) / self._scale_const)
        # K_nu returns nan at zero
        return where(isnan(result), 1, result)


class GaussianCorrelation(DistanceCorrelationFunction):
    """A gaussian correlation structure.

    Notes
    -----
    Correlation given by exp(-dist**2 / (length**2)) where dist is the
    distance between the points.
    """

    def __call__(self, dist):
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        scaled_dist2 = square(dist / self._length)
        return exp(-scaled_dist2)
