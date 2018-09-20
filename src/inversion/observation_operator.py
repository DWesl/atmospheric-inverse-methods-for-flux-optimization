"""Utilities for working with influence function files.

Designed for files dimensioned:
`observation_time, site, forecast_period, y, x`.

Functions here turn these into BSR matrices dimensioned
`observation, time, y, x` or
`observation_time, site, time, y, x`
keeping the memory savings of the first form while still usable for
standard code.
"""
import numpy as np
from numpy import newaxis

import scipy.sparse
import pandas as pd

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
    flux_time_index = pd.date_range(
        earliest_flux.values,
        flux_time.max().values,
        freq="{interval:d}S".format(
            interval=int(
                dt.values / np.timedelta64(1, "s")
            )
        ))

    # Find offset of first flux in each row
    row_offset_start = np.asarray(
        (flux_time[:, -1] - earliest_flux) / dt
    ).astype(np.int64)

    # repeat row_offset_start for sites
    y_index = obs_op.get_index("dim_y")
    x_index = obs_op.get_index("dim_x")

    # obs_op.stack(observation=("observation_time", "site"))
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

    aligned_ds = xarray.DataArray(
        aligned_data,
        coords=dict(
            observation=(
                xarray.core.utils.multiindex_from_product_levels(
                    [obs_op.indexes["observation_time"],
                     obs_op.indexes["site"]],
                    names=["forecast_reference_time", "site"]
                )
            ),
            flux_state_space=(
                xarray.core.utils.multiindex_from_product_levels(
                    [flux_time_index,
                     y_index,
                     x_index],
                    names=["time", "dim_y", "dim_x"]
                )
            ),
        ),
        dims=("observation", "flux_state_space"),
        name=obs_op.name,
        attrs=obs_op.attrs,
        encoding=obs_op.encoding,
    )

    for coord_name in obs_op.coords:
        # I already have some coords
        # Coords for dimensions don't carry over
        # I've already taken care of time dims
        if (((coord_name not in aligned_ds.coords and
              coord_name not in obs_op.indexes) and
             "time" not in coord_name)):
            aligned_ds.coords[coord_name] = obs_op.coords[coord_name]
    return aligned_ds


def align_partial_obs_op(obs_op):
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

    x_index_name = [name for name in obs_op.dims if "x" in name][0]
    y_index_name = [name for name in obs_op.dims if "y" in name][0]
    flux_time_index = pd.date_range(
        earliest_flux.values,
        flux_time.max().values,
        freq="{interval:d}S".format(
            interval=int(
                dt.values / np.timedelta64(1, "s")
            )
        ))
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
    ).stack(
        block_dim=("observation", "time_before_observation")
    ).expand_dims(
        "block_extra_dim", 1
    ).transpose(
        "block_dim", "block_extra_dim", "space"
    )

    aligned_data = scipy.sparse.bsr_matrix(
        (data.data,
         (row_offset_start[:, newaxis] + column_offset[newaxis, :]).flat,
         np.arange(flux_time.shape[0] + 1) * flux_time.shape[1]))
    # rows: obs_time, site stack
    # cols: flux_time, space stack
    aligned_data.ndim = 2

    col_index = xarray.core.utils.multiindex_from_product_levels(
        (flux_time_index,
         y_index, x_index),
        ("flux_time", y_index_name, x_index_name))

    aligned_ds = xarray.DataArray(
        aligned_data,
        dict(
            observation=obs_op.indexes["observation"],
            flux_state_space=col_index,
        ),
        ("observation", "flux_state_space"),
        obs_op.name,
        obs_op.attrs,
        obs_op.encoding,
    )
    for coord_name in obs_op.coords:
        # I already have some coords
        # Dim coords don't carry over
        if ((coord_name not in aligned_ds.coords and
             coord_name not in obs_op.indexes and
             "time" not in coord_name)):
            aligned_ds.coords[coord_name] = obs_op.coords[coord_name]
    return aligned_ds
            
