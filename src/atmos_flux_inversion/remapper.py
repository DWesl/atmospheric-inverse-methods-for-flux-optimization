"""Helpers for reducing the resolution of various operators.

Intended for use in calculating the uncertainties at a coarser
resolution than the fluxes.
"""
from __future__ import division
from math import ceil

import numpy as np
from numpy import zeros, newaxis
from numpy.linalg import inv

DTYPE = np.float64


def get_remappers(domain_size, block_side=3):
    """Get matrices to remap from original to coarser resolution.

    Parameters
    ----------
    domain_size: Tuple[int, int]
       The size of the spatial domain.
    block_side: int, optional
       The number of original cells in each direction to combine into
       a single new cell.

    Returns
    -------
    extensive_remapper: array_like
        A matrix for changing extensive quantities or finding the sum.
        In this package, that would be the observation operators.
    intensive_remapper: array_like
        A matrix for changing intensive quantities or finding the mean.
        In this package, that would be the fluxes.
    """
    domain_size = tuple(domain_size)
    reduced_size = tuple(int(ceil(dim / block_side)) for dim in domain_size)
    extensive_remapper = zeros(reduced_size + domain_size,
                               dtype=DTYPE)

    for i in range(reduced_size[0]):
        old_i = block_side * i
        for j in range(reduced_size[1]):
            old_j = block_side * j
            extensive_remapper[i, j, old_i:old_i + block_side,
                               old_j:old_j + block_side] = 1

    assert old_i + block_side >= domain_size[0] - 1
    assert old_j + block_side >= domain_size[1] - 1
    intensive_remapper = extensive_remapper.copy()
    n_nz = intensive_remapper.sum(axis=(-1, -2))
    intensive_remapper /= n_nz[:, :, newaxis, newaxis]

    return extensive_remapper, intensive_remapper


def get_optimal_prolongation(reduction, covariance):
    """Find the optimal prolongation for the given reduction.

    The optimal prolongation operator depends on both the reduction
    operator used and the covariance function assumed:

    .. math:: pro = cov @ red^T @ (red @ cov @ red^T)^{-1}

    This prolongation minimizes the aggregation error for the
    inversion with the given reduction.

    Parameters
    ----------
    reduction: array_like
    covariance: LinearOperator

    Returns
    -------
    prolongation: array_like

    Notes
    -----
    The transpose of the `extensive` remapper from
    :func:`get_remappers` will function as a prolongation operator and
    corresponds to `covariance` being a multiple of the identity
    matrix.

    References
    ----------
    Bousserez, N. and D. Henze "Optimal and scalable methods to
    approximate the solutions of large-scale Bayesian problems: theory
    and application to atmospheric inversions and data assimilation"
    *Quarterly Journal of the Royal Meteorological Society*
    2018. Vol. 144, pp. 365--390. :doi:`10.1002/qj.3209`
    """
    cov_dot_red_T = covariance.dot(reduction.T)
    return cov_dot_red_T.dot(
        inv(reduction.dot(cov_dot_red_T))
    )
