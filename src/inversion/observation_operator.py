"""Utilities for working with influence function files.

Designed for files dimensioned:
`observation_time, site, forecast_period, y, x`.

Functions here turn these into BSR matrices dimensioned
`observation, time, y, x` or
`observation_time, site, time, y, x`
keeping the memory savings of the first form while still usable for
standard code.

I should mention at some point that "influence function" and
"observation operator" refer to the same quantity, represented by
:math:`H` in the equations.  This is used to represent which fluxes
influence each observation, as well as what observations would be
expected given a specific set of fluxes.
"""
import numpy as np
from numpy import newaxis

import scipy.sparse

import xarray.core.utils
import xarray.core.variable

xarray.core.variable.NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    xarray.core.variable.NON_NUMPY_SUPPORTED_ARRAY_TYPES +
    (scipy.sparse.spmatrix,)
)


def align_full_obs_op(obs_op):
    """Align observation operator on flux_time.

    Parameters
    ----------
    obs_op: xarray.DataArray
        dims: observation_time, site, time_before_observation, y, x
        coords: flux_time
        time_before_obs monotone increasing
        observation_time monotone decreasing

    Returns
    -------
    obs_op_aligned: xarray.DataArray
        dims: observation, flux
        MultiIndices:
            observation_time, site
            flux_time, y, x
        data backed by scipy.sparse.bsr_matrix
        bsr_matrix does not support indexing
    """
    flux_time = obs_op.coords["flux_time"]

    column_offset = np.arange(flux_time.shape[1])[::-1]
    dt = abs(flux_time[0, 1] - flux_time[0, 0])
    earliest_flux = flux_time.min()

    # Find offset of first flux in each row
    row_offset_start = np.asarray(
        (flux_time[:, -1] - earliest_flux) / dt
    ).astype(np.int64)

    # repeat row_offset_start for sites
    obs_op = obs_op.stack(space=("dim_y", "dim_x"))
    data = obs_op.transpose(
        "observation_time", "time_before_observation",
        "site", "space"
    ).stack(
        block_dim=("observation_time", "time_before_observation")
    ).transpose(
        "block_dim", "site", "space"
    )

    aligned_data = scipy.sparse.bsr_matrix(
        (data.data,
         (row_offset_start[:, newaxis] +
          column_offset[newaxis, :]).reshape(-1),
         np.arange(flux_time.shape[0] + 1) * flux_time.shape[1]))
    # rows: obs_time, site stack
    # cols: flux_time, space stack
    aligned_data.ndim = 2

    return aligned_data


def align_partial_obs_op(obs_op, required_shape=None):
    """Align observation operator on flux_time.

    Parameters
    ----------
    obs_op: xarray.DataArray
        dims: observation, time_before_obs, y, x
        coords: flux_time, observation_time
        time_before_obs monotone increasing
        observation_time monotone decreasing
        only one dim each with 'y' and 'x'
        obs_op[-1, -1] is earliest influence map
    required_shape: tuple of int, optional
        The share required of the returned
        matrix.

    Returns
    -------
    obs_op_aligned: xarray.DataArray
        dims: observation, flux
        MultiIndices:
            observation_time, site
            flux_time, y, x
        data backed by scipy.sparse.bsr_matrix
        bsr_matrix does not support indexing
    """
    flux_time = obs_op.coords["flux_time"]

    column_offset = np.arange(flux_time.shape[1])[::-1]
    dt = abs(flux_time[0, 1] - flux_time[0, 0])
    earliest_flux = flux_time.min()

    y_index_name = [name for name in obs_op.dims if "y" in name][0]
    x_index_name = y_index_name.replace("y", "x")
    x_index = obs_op.get_index(x_index_name)
    y_index = obs_op.get_index(y_index_name)

    # Find offset of first flux in each row
    row_offset_start = np.asarray(
        (flux_time[:, -1] - earliest_flux) / dt
    ).astype(np.int64)

    # repeat row_offset_start for sites

    # obs_op.stack(observation=("observation_time", "site"))
    obs_op = obs_op.stack(space=(y_index_name, x_index_name))
    data = obs_op.transpose(
        "observation", "time_before_observation",
        "space"
    ).expand_dims(
        "block_extra_dim", 1
    ).transpose(
        "observation", "time_before_observation", "block_extra_dim", "space"
    )
    n_obs = len(obs_op.indexes["observation"])
    n_times_back = len(obs_op.indexes["time_before_observation"])
    n_space = len(y_index) * len(x_index)

    aligned_data = scipy.sparse.bsr_matrix(
        (data.data.reshape(n_obs * n_times_back, 1, n_space),
         (row_offset_start[:, newaxis] + column_offset[newaxis, :]).flat,
         np.arange(flux_time.shape[0] + 1) * flux_time.shape[1]),
        shape=required_shape)
    # rows: obs_time, site stack
    # cols: flux_time, space stack
    aligned_data.ndim = 2

    return aligned_data
