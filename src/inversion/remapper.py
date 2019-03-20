"""Helpers for reducing the resolution of various operators.

Intended for use in calculating the uncertainties at a coarser
resolution than the fluxes.
"""
from __future__ import division
from math import ceil

import numpy as np
from numpy import zeros, newaxis, arange
from numpy import sum as np_sum

from xarray import DataArray

DTYPE = np.float64


def get_spatial_remappers(domain_size, block_side=3):
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


def remap_bsr_temporal(flux_time_index, new_freq, matrix_to_remap):
    """Remap a BSR matrix using space as the blocks.

    It is assumed that the blocks are n by n_flux_points_space,
    and that the block column index represents time.

    If you have a full array you want on a different temporal
    frequency, attach metadata to make it a :class:`~xarray.DataArray`
    and use its :meth:`~xarray.DataArray.resample`.

    Parameters
    ----------
    flux_time_index: pandas.DatetimeIndex
    new_freq: str
    matrix_to_remap: scipy.sparse.bsr_matrix

    Returns
    -------
    remapped_matrix: array_like

    Raises
    ------
    ValueError
        If new_freq is specified in terms of weeks.  "7D" is fine.
        "1W" uses different logic.
    """
    if new_freq.endswith("W"):
        raise ValueError(
            "Weekly frequencies are weird.  I don't support them here.")

    row_starts = matrix_to_remap.indptr
    columns = matrix_to_remap.indices
    data = matrix_to_remap.data
    blocksize = matrix_to_remap.blocksize
    n_flux_times = len(flux_time_index)

    if matrix_to_remap.shape[1] / blocksize[1] != n_flux_times:
        raise ValueError("Index does not correspond to structure passed.")

    test_series = DataArray(
        data=arange(len(flux_time_index)),
        coords=dict(time=flux_time_index.values),
        dims="time",
        name="temp_array",
    )
    resampled = test_series.resample(time=new_freq).sum("time")
    resampled_index = resampled.indexes["time"]

    result = zeros((matrix_to_remap.shape[0],
                    len(resampled),
                    blocksize[1]),
                   # The reduced observation operator is used in:
                   #   H.T @ vec
                   #   la.solve(mat, H)
                   # In both cases H being F-contiguous is an
                   # advantage.
                   order="F",
                   dtype=matrix_to_remap.dtype)

    # columns[new_i] is list of old_i that get summed to make
    # result[:, new_i]
    columns_old_to_new = [
        [old_i for old_i in range(len(test_series))
         if (resampled_index[new_i] <=
             flux_time_index[old_i] <
             resampled_index[new_i + 1])]
        for new_i in range(0, len(resampled_index) - 1)]
    if columns_old_to_new:
        columns_old_to_new.append(
            range(columns_old_to_new[-1][-1] + 1, len(flux_time_index)))
    else:
        columns_old_to_new.append(
            range(len(flux_time_index)))

    # The full observation operator, however, would almost certainly
    # be C-contiguous.  Since that would likely dominate the time,
    # have the outer loop be over rows here.
    for block_i in range(result.shape[0] // blocksize[0]):
        list_of_columns = columns[
            row_starts[block_i]:row_starts[block_i + 1]
        ]
        for j in range(len(resampled)):
            columns_in_block = [
                old_j for old_j in list_of_columns
                if old_j in columns_old_to_new[j]
            ]
            result_i = block_i * blocksize[0]
            np_sum(
                data[columns_in_block, :, :],
                axis=0,
                out=result[result_i:result_i + blocksize[0],
                           j, :],
            )

    return result.reshape(
        matrix_to_remap.shape[0],
        len(resampled) * blocksize[1]
    )
