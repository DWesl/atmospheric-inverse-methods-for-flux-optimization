#!/usr/bin/env python
"""Check the LPDM transport against the WRF transport.

The two models should give similar results at tower locations when
given similar fluxes.  This script produces plots to check that.
"""
from __future__ import print_function, division, unicode_literals

import argparse
import datetime
import logging
import os.path

import dask.config as dask_conf
import dateutil.tz
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import pandas as pd
import pint
import pyproj
import sparse
import xarray
import wrf

TYPING = False
if TYPING:
    import matplotlib as mpl
    from typing import Dict, Any, Hashable

logging.basicConfig(
    format=(
        "%(asctime)s:%(levelname)7s:%(name)8s:"
        "%(module)20s:%(funcName)15s:%(lineno)03s: %(message)s"
    ),
    level=logging.DEBUG,
)
_LOGGER = logging.getLogger(__name__)
_LOGGER.info("Done imports, starting code")

dask_conf.set(num_workers=8, scheduler="threads")
xarray.set_options(display_width=100, keep_attrs=True)

# Physical constants
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
UTC = dateutil.tz.tzutc()

# Global unit registry
UREG = pint.UnitRegistry()
UREG.define("[mole fraction] = [number] / [number]")
UREG.define("ppm = micromole / mole")
UREG.define("@alias ppm = ppmv")

FLUX_INTERVAL = 3
"""The interval at which fluxes become available in hours.

Fluxes are usually integrated forward from hourly input, but I can't
solve for that in a reasonable timeframe.

This determines both the input and the output time resolution as well
as what the inversion solves for.

Note
----
Must divide twenty-four.
"""
FLUX_WINDOW = HOURS_PER_DAY * DAYS_PER_WEEK * 3
"""How long fluxes considered to have an influence.

Measured in hours.
Implemented by slicing out this much of the stored influence functions
for use as the linearized observation operator in the inversion.
"""
FLUX_RESOLUTION = 27
"""Resolution of fluxes and influence functions in kilometers."""
N_GROUPS = 6
"""Number of groups of towers LPDM runs."""


def make_sparse_ds(lpdm_footprint_ds):
    # type: (xarray.Dataset) -> xarray.Dataset
    """Finish formatting a sparse dataset.

    Intended to be used as a callback in xarray.open_dataset and
    friends.

    Parameters
    ----------
    lpdm_footprint_ds: xarray.Dataset

    Returns
    -------
    transformed_lpdm_footprint_ds: xarray.Dataset

    Note
    ----
    Will load lpdm_footprint_ds into memory to make it sparse.

    """
    _LOGGER.debug("Loading data for file %s", lpdm_footprint_ds.encoding["source"])
    lpdm_footprint_ds = lpdm_footprint_ds.load()
    _LOGGER.debug(
        "Making sparse influence function for file %s",
        lpdm_footprint_ds.encoding["source"],
    )
    # Find the list of axes
    # To make this truly general, we'd need to check for {name}_values
    # {name}_coords pairs in lpdm_footprint_ds.data_vars.  I'm not doing that yet,
    # since I'm only using it here.
    if "compress" not in lpdm_footprint_ds["H_values"].attrs:
        lpdm_footprint_ds["H_values"].attrs["compress"] = [
            "observation_time",
            "site",
            "time_before_observation",
            "dim_y",
            "dim_x",
        ]
    elif isinstance(lpdm_footprint_ds["H_values"].attrs["compress"], str):
        lpdm_footprint_ds["H_values"].attrs["compress"] = (
            lpdm_footprint_ds["H_values"].attrs["compress"].split()
        )
    # Create the actual sparse DataArray
    sparse_data = sparse.COO(
        lpdm_footprint_ds["H_coords"].values,
        lpdm_footprint_ds["H_values"].values,
        shape=tuple(
            lpdm_footprint_ds.dims[dim]
            for dim in lpdm_footprint_ds["H_values"].attrs["compress"]
        ),
    )
    lpdm_footprint_ds["H"] = (
        lpdm_footprint_ds["H_values"].attrs["compress"],
        sparse_data,
        {
            key: value
            for key, value in lpdm_footprint_ds["H_values"].attrs.items()
            if key != "compress"
        },
    )
    _LOGGER.debug("Made sparse influence function, fixing coords")
    del lpdm_footprint_ds["H_values"], lpdm_footprint_ds["H_coords"]
    obs_time_index = lpdm_footprint_ds.indexes["observation_time"].round("S")
    lpdm_footprint_ds.coords["observation_time"] = obs_time_index
    time_back_index = lpdm_footprint_ds.indexes["time_before_observation"].round("S")
    lpdm_footprint_ds.coords["time_before_observation"] = time_back_index
    lpdm_footprint_ds.coords["site"] = np.char.decode(
        lpdm_footprint_ds["site_names"].values, "ascii"
    )
    return lpdm_footprint_ds


def read_wrf_file(wrf_name):
    # type: (str) -> xarray.Dataset
    """Set coordinates on the WRF dataset.

    Which variables I set as coordinates is chosen mostly by personal
    preference.

    Parameters
    ----------
    wrf_name: str

    Returns
    -------
    wrf_ds
    """
    with netCDF4.Dataset(wrf_name, "r") as wrf_nc:
        height_agl = wrf.getvar(wrf_nc, "height_agl", squeeze=False)
        time = wrf.getvar(wrf_nc, "times")
    wrf_ds = xarray.open_dataset(wrf_name, chunks={"Time": 1}).set_coords(
        [
            # Dim coords
            "ZNU",
            "ZNW",
            "ZS",
            "XTIME",
            # formula terms
            "P_TOP",
            "PSFC",
            # Ancillary vars
            "HGT",
            "XLAND",
            "VEGFRA",
            # Technically data vars, but I need them to interpret
            # the rest properly.
            "PH",
            "PHB",
            "P",
            "PB",
        ]
    )
    # Some fraction of these don't have time as a dim to start with.
    # # These variables don't change in static nesting scenarios
    # for coord_name in (
    #     "ZNU",
    #     "ZNW",
    #     "ZS",
    #     "P_TOP",
    #     "HGT",
    #     "XLAT",
    #     "XLONG",
    # ):
    #     wrf_ds.coords[coord_name] = wrf_ds.coords[coord_name].isel(Time=0)
    wrf_ds.coords["height_agl"] = height_agl
    wrf_ds.coords["wrf_proj"] = -1
    wrf_ds.coords["wrf_proj"].attrs.update(height_agl.attrs["projection"].cf())
    wrf_ds.coords["wrf_proj"].attrs["wkt"] = pyproj.Proj(
        **height_agl.attrs["projection"].cartopy().proj4_params
    ).crs.to_wkt()
    del height_agl.attrs["projection"]
    wrf_ds.coords["Time"] = (
        ("Time",),
        time.expand_dims("Time"),
        {"standard_name": "time", "calendar": "standard"},
    )
    return wrf_ds


def get_lpdm_footprint(lpdm_footprint_dir, year, month):
    # type: (str, int, int) -> xarray.Dataset
    """Read in LPDM footprints for a month.

    Parameters
    ----------
    lpdm_footprint_dir: str
    year: int
    month: int

    Returns
    -------
    lpdm_footprint: xarray.Dataset
    """
    influence_files = [
        os.path.join(
            lpdm_footprint_dir,
            "{0:04d}".format(year),
            "{0:02d}".format(month),
            "GROUP{0:1d}".format(group + 1),
            (
                "LPDM_{year:04d}_{month:02d}_{group:02d}_"
                "{flux_interval:02d}hrly_{res:03d}km_molar_footprints.nc4"
            ).format(
                year=year,
                month=month,
                group=group,
                flux_interval=FLUX_INTERVAL,
                res=FLUX_RESOLUTION,
            ),
        )
        for group in range(N_GROUPS)
    ]
    _LOGGER.debug("Influence Files: [\n\t%s\n]", "\n\t".join(influence_files))
    influence_datasets = []
    for name in influence_files:
        _LOGGER.debug("Reading influences from file %s", name)
        influence_datasets.append(
            xarray.open_dataset(
                name, chunks={"observation_time": 1, "site": 1}
            ).set_index(site="site_names")
            # .rename(site="site_names")
        )
        _LOGGER.debug("Done reading file %s", name)
    _LOGGER.debug("Concatenating influence functions into single dataset")
    influence_dataset = xarray.concat(
        influence_datasets,
        dim="site",
    )
    _LOGGER.debug("Alphabetizing towers in influence functions")
    influence_dataset = influence_dataset.reindex(
        site=sorted(influence_dataset.indexes["site"])
    )
    _LOGGER.debug("Influence dataset:\n%s", influence_dataset)
    _LOGGER.debug("Aligning influence functions on flux time")
    obs_time_index = influence_dataset.indexes["observation_time"]
    # first_obs_time = min(obs_time_index)
    # last_obs_time = max(obs_time_index)
    # flux_start = (
    #     first_obs_time - max(influence_dataset.indexes["time_before_observation"])
    # ).replace(hour=0)
    # if last_obs_time.hour != 0:
    #     flux_end = last_obs_time.replace(hour=0) + datetime.timedelta(days=1)
    # else:
    #     flux_end = last_obs_time
    # flux_time_index = pd.date_range(
    #     flux_start,
    #     flux_end,
    #     freq="{flux_interval:d}H".format(flux_interval=FLUX_INTERVAL),
    #     tz="UTC",
    #     closed="right",
    #     name="flux_times",
    # )
    aligned_influence = xarray.concat(
        [
            influence_dataset["H"]
            .isel(observation_time=i)
            .set_index(time_before_observation="flux_time")
            .rename({"time_before_observation": "flux_time"})
            .astype(np.float32)
            for i in range(len(obs_time_index))
        ],
        dim="observation_time",
        fill_value=np.array(0, dtype=influence_dataset["H"].dtype),
    ).to_dataset()
    _LOGGER.debug("Rechunking aligned dataset")
    aligned_influence = aligned_influence.chunk(
        {"flux_time": 8, "observation_time": 24}
    )
    _LOGGER.debug("Adding bounds and coords to influence functions")
    for bound_name in (
        "observation_time_bnds",
        "dim_y_bnds",
        "dim_x_bnds",
        "wrf_proj",
        "flux_time_bnds",
        "height_bnds",
        "lpdm_configuration",
        "wrf_configuration",
    ):
        aligned_influence[bound_name] = influence_dataset[bound_name]
    aligned_influence.attrs["history"] = (
        "{0:s}: Influence functions for a month aligned on flux time and "
        "combined into one file\n{1:s}"
    ).format(
        datetime.datetime.utcnow().isoformat(), influence_datasets[0].attrs["history"]
    )
    aligned_influence.attrs["file_list"] = " ".join(influence_files)
    aligned_influence.attrs.update(influence_datasets[0].attrs)
    aligned_influence["H"].attrs.update(influence_datasets[0]["H"].attrs)
    return aligned_influence


def save_sparse_influences(lpdm_footprint, save_name):
    # type: (xarray.Dataset, str) -> None
    """Save the sparse influence functions in a sparse format.

    Parameters
    ----------
    lpdm_footprint: xarray.Dataset
    save_name: str
    """
    _LOGGER.debug("Store sparse format data directly in dataset")
    sparse_infl_fun = sparse.COO(lpdm_footprint["H"].data)
    lpdm_footprint["H_coords"] = (
        ("H_coo_coords", "H_coo_nnz"),
        # I know int16 will work for a single month
        # If saving footprints for mu
        sparse_infl_fun.coords.astype(np.int16),
        {
            "description": (
                "Indexes of nonzero values in a multidimensional array, "
                "as used in a COO matrix."
            ),
        },
    )
    lpdm_footprint["H_coords"].encoding.update({"zlib": True})
    lpdm_footprint["H_values"] = (
        ("H_coo_nnz",),
        sparse_infl_fun.data,
        {
            "description": (
                "Nonzero values in a multidimensional array, "
                "as used in a COO matrix."
            ),
        },
    )
    _LOGGER.debug("Update coordinates, attributes, and encoding")
    lpdm_footprint["H_values"].attrs.update(lpdm_footprint["H"].attrs)
    lpdm_footprint["H_values"].encoding.update({"zlib": True})
    lpdm_footprint["H_coo_nnz"] = (
        ("H_coo_nnz",),
        np.ravel_multi_index(sparse_infl_fun.coords, lpdm_footprint["H"].shape),
        {
            "compress": " ".join(str(dim) for dim in lpdm_footprint["H"].dims),
            "description": "Indices of nonzero values in flattened array",
        },
    )
    del lpdm_footprint["H"]
    encoding = {
        name: {"zlib": True, "_FillValue": -9.99e9} for name in lpdm_footprint.data_vars
    }  # type: Dict[Hashable, Dict[str, Any]]
    encoding.update(
        {name: {"zlib": True, "_FillValue": None} for name in lpdm_footprint.coords}
    )
    lpdm_footprint.to_netcdf(save_name, mode="w", format="NETCDF4")


def get_lpdm_tower_locations(lpdm_footprint):
    # type: (xarray.Dataset) -> xarray.DataArray
    """Find the tower locations a footprint is good for.

    Parameters
    ----------
    lpdm_footprint: xarray.Dataset

    Returns
    -------
    locations: xarray.DataArray
    """
    return lpdm_footprint.coords["site"]


def get_wrf_fluxes(wrf_output_dir, year, month):
    # type: (str, int, int) -> xarray.Dataset
    """Get WRF fluxes for a given month.

    Parameters
    ----------
    wrf_output_dir: str
    year: int
    month: int

    Returns
    -------
    fluxes: xarray.Dataset
    """
    flux_time_index = pd.date_range(
        start=datetime.datetime(year, month, 1, 0, 0, 0)
        - datetime.timedelta(hours=FLUX_WINDOW),
        end=datetime.datetime(year, month + 1, 1, 0, 0, 0),
        freq="{0:d}H".format(FLUX_INTERVAL),
    )
    wrf_output_files = [
        os.path.join(
            wrf_output_dir, "wrfout_d01_{date:%Y-%m-%d_%H:%M:%S}".format(date=date)
        )
        for date in flux_time_index
    ]
    _LOGGER.debug("Reading WRF files for fluxes")
    flux_datasets = []
    for name in wrf_output_files:
        _LOGGER.debug("Reading fluxes from file %s", name)
        wrf_ds = read_wrf_file(name)
        flux_names = [
            name for name in wrf_ds.data_vars if str(name).startswith("E_TRA")
        ]
        flux_datasets.append(wrf_ds[flux_names])
    _LOGGER.debug("Combining fluxes into single dataset")
    flux_dataset = xarray.concat(flux_datasets, dim="Time")
    flux_dataset.coords["Time"] = (
        ("Time",),
        flux_time_index,
        {"standard_name": "time"},
    )
    flux_dataset.attrs[
        "history"
    ] = "{0:s}: WRF fluxes combined into single file\n{1:s}".format(
        datetime.datetime.utcnow().isoformat(),
        flux_datasets[0].attrs.get("history", ""),
    )
    flux_dataset.attrs["file_list"] = " ".join(wrf_output_files)
    _LOGGER.debug("Returned flux dataset:\n%s", flux_dataset)
    return flux_dataset


def get_wrf_mole_fractions(wrf_output_dir, year, month, tower_locs):
    # type: (str, int, int, xarray.DataArray) -> xarray.Dataset
    """Extract the WRF mole fractions at the tower locations.

    Parameters
    ----------
    wrf_output_dir: str
    year: int
    month: int
    tower_locs: xarray.Dataset

    Returns
    -------
    mole_fraction_time_series: xarray.Dataset
    """
    observation_time_index = pd.date_range(
        start=datetime.datetime(year, month, 1, 0, 0, 0),
        end=datetime.datetime(year, month + 1, 1, 0, 0),
        # The one place where OBS_WINDOW/OBSERVATION_WINDOW is
        # relevant here
        freq="1H",
    )
    wrf_output_files = [
        os.path.join(
            wrf_output_dir, "wrfout_d01_{date:%Y-%m-%d_%H:%M:%S}".format(date=date)
        )
        for date in observation_time_index
    ]
    with netCDF4.Dataset(wrf_output_files[0], "r") as reference_ds:
        # ll_to_ji would be a better name
        east_index, south_index = wrf.ll_to_xy(
            reference_ds, tower_locs["site_lats"], tower_locs["site_lons"]
        )
    del reference_ds
    _LOGGER.debug("Tower locations and coordinates: %s", tower_locs)
    _LOGGER.debug("East and south indices:\n%s\n%s", east_index, south_index)
    # east_index.coords["idx"] = tower_locs.coords["site"]
    # south_index.coords["idx"] = tower_locs.coords["site"]
    wrf_mole_fractions = []
    for name in wrf_output_files:
        wrf_ds = read_wrf_file(name)
        mole_fraction_names = [
            name for name in wrf_ds.data_vars if str(name).startswith("tracer_")
        ]
        mole_fraction_fields = wrf_ds[mole_fraction_names]
        wrf_mole_fractions.append(
            mole_fraction_fields.isel(south_north=south_index, west_east=east_index)
        )
    result = xarray.concat(wrf_mole_fractions, dim="Time").rename(idx="site")
    for tower_coord_name in ("site", "site_lats", "site_lons"):
        result.coords[tower_coord_name] = tower_locs.coords[tower_coord_name]
    # Now redundant.
    del result.coords["latlon_coord"]
    _LOGGER.debug("WRF mole fractions: %s", result)
    return result


def lpdm_footprint_convolve(lpdm_footprint, wrf_fluxes):
    # type: (xarray.Dataset, xarray.Dataset) -> xarray.Dataset
    """Convolve the footprint with the fluxes.

    Parameters
    ----------
    lpdm_footprint: xarray.Dataset
    wrf_fluxes: xarray.Dataset

    Returns
    -------
    lpdm_mole_fractions: xarray.Dataset
    """
    fluxes_matched = wrf_fluxes.rename(
        Time="flux_time", west_east="dim_x", south_north="dim_y"
    ).sum("emissions_zdim")
    result = xarray.Dataset()
    for i in range(len(wrf_fluxes.data_vars)):
        _LOGGER.debug("Influence function to convolve:\n%s", lpdm_footprint["H"])
        _LOGGER.debug(
            "Fluxes to convolve:\n%s", fluxes_matched["E_TRA{i:d}".format(i=i + 1)]
        )
        here_fluxes = fluxes_matched["E_TRA{i:d}".format(i=i + 1)]
        result["tracer_{i:d}".format(i=i + 1)] = lpdm_footprint["H"].dot(here_fluxes)
        # I really don't understand why this crashes
        # TODO: fix crashes in code below
        result["tracer_{i:d}".format(i=i + 1)].attrs.update(
            {
                "standard_name": "carbon_dioxide_mole_fraction",
                "long_name": (
                    "carbon_dioxide_mole_fraction_enhancement_tracer_{0:d}".format(
                        i + 1
                    )
                ),
                "units": str(
                    UREG(lpdm_footprint["H"].attrs["units"])
                    * UREG(here_fluxes.attrs["units"])
                ),
                "description": (
                    "CO2 mole fractions predicted by LPDM for tracer {0:d}".format(
                        i + 1
                    )
                ),
            }
        )
    return result


def compare_wrf_lpdm_mole_fractions_for_month(
    wrf_mole_fractions, lpdm_mole_fractions, year, month
):
    # type: (xarray.Dataset, xarray.Dataset, int, int) -> mpl.figure.Figure
    """Plot the WRF and LPDM mole fractions at the LPDM towers.

    Parameters
    ----------
    wrf_mole_fractions, lpdm_mole_fractions: xarray.Dataset
    year, month: int
    """
    combined_mole_fractions = xarray.concat(
        [
            wrf_mole_fractions.isel(bottom_top=5).rename(Time="observation_time"),
            lpdm_mole_fractions,
        ],
        dim=pd.Index(["WRF", "LPDM"], name="model"),
    )

    for tracer_name in combined_mole_fractions.data_vars:
        tracer_num = int(tracer_name.split("_")[1])
        if tracer_num == 1:
            wrf_background = 400
        elif tracer_num in range(2, 12 + 1):
            wrf_background = 300
        else:
            wrf_background = 0
        fig, axes = plt.subplots(7, 5, figsize=(9, 6.5), sharex=True, sharey=True)
        fig.autofmt_xdate()
        fig.subplots_adjust(top=0.96, bottom=0.1, left=0.1, right=0.95, hspace=0.6)
        fig.suptitle(
            "WRF and LPDM mole fractions for {0:04d}-{1:02d}\nTracer {2:d}".format(
                year, month, tracer_num
            )
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for site, ax in zip(
            combined_mole_fractions.coords["site"].values[::-1], axes.flat[::-1]
        ):
            ax.set_visible(True)
            # combined_mole_fractions[tracer_name].sel(site=site).plot.line(
            #     x="observation_time", hue="model", ax=ax
            # )
            wrf_line = ax.plot(
                combined_mole_fractions.coords["observation_time"].values,
                combined_mole_fractions[tracer_name].sel(model="WRF", site=site).values
                - wrf_background,
            )[0]
            lpdm_line = ax.plot(
                combined_mole_fractions.coords["observation_time"].values,
                combined_mole_fractions[tracer_name].sel(model="LPDM", site=site).values
                / 1000,
            )[0]
            ax.set_title(site.decode("utf8"))
            ax.set_xticks(
                mdates.date2num(
                    pd.date_range(
                        datetime.datetime(year, month, 1),
                        datetime.datetime(year, month + 1, 1),
                        freq="7D",
                    ).values
                )
            )
        for ax in axes[:, 0]:
            ax.set_ylabel("CO\N{SUBSCRIPT TWO} tracer\n(ppm)")
        fig.legend(
            [wrf_line, lpdm_line],
            ["WRF (\N{APPROXIMATELY EQUAL TO}200m AGL)", "LPDM"],
            loc="upper left",
        )
        fig.savefig(
            "wrf-lpdm-mole-fraction-comparison-{0:03d}-{1:02d}-tracer-{2:d}.pdf".format(
                year, month, tracer_num
            )
        )
        # PNG is slightly bigger.  Saving as 4-color would help, if I know how.
        # fig.savefig(
        #     "wrf-lpdm-mole-fraction-comparison-{0:03d}-{1:02d}-tracer-{2:d}.png".format(
        #         year, month, tracer_num
        #     )
        # )
        plt.close(fig)
    return


def save_nonsparse_netcdf(ds_to_save, save_name):
    # type: (xarray.Dataset, str) -> None
    """Save the dataset at the name.

    Parameters
    ----------
    ds_to_save: xarray.Dataset
    save_name: str
    """
    # I may need to rethink the fill value with integer datasets
    encoding = {
        name: {"zlib": True, "_FillValue": -9.99e9} for name in ds_to_save.data_vars
    }  # type: Dict[Hashable, Dict[str, Any]]
    encoding.update(
        {name: {"zlib": True, "_FillValue": None} for name in ds_to_save.coords}
    )
    ds_to_save.to_netcdf(save_name, mode="w", format="NETCDF4", encoding=encoding)


def existing_directory(path):
    # type: (str) -> str
    """Check whether the path is an existing directory.

    Parameters
    ----------
    path: str

    Returns
    -------
    path: str

    Raises
    ------
    argparse.ArgumentTypeError
        If path is not an existing directory
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            "Directory '{0:s}' does not exist".format(path)
        )
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("'{0:s}' is not a directory".format(path))
    return path


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("lpdm_footprint_dir", type=existing_directory)
PARSER.add_argument("wrf_output_dir", type=existing_directory)
PARSER.add_argument("output_dir", type=existing_directory)
PARSER.add_argument("year", type=int)
PARSER.add_argument("month", type=int)

if __name__ == "__main__":
    _LOGGER.info("Done definitions, starting execution")
    args = PARSER.parse_args()
    _LOGGER.debug("Command-line arguments: %s", args)
    lpdm_footprint = get_lpdm_footprint(args.lpdm_footprint_dir, args.year, args.month)
    _LOGGER.info("Footprints read")
    lpdm_locs = get_lpdm_tower_locations(lpdm_footprint).load()
    _LOGGER.info("Have locations")
    wrf_fluxes = get_wrf_fluxes(args.wrf_output_dir, args.year, args.month)
    _LOGGER.info("Have fluxes")
    wrf_mole_fractions = get_wrf_mole_fractions(
        args.wrf_output_dir, args.year, args.month, lpdm_locs
    )
    _LOGGER.info("Have WRF mole fractions")
    wrf_mole_fractions = wrf_mole_fractions.load()
    _LOGGER.info("Loaded WRF mole fractions")
    _LOGGER.debug("About to load influence functions")
    lpdm_footprint = lpdm_footprint.persist()
    _LOGGER.info("Loaded influence functions")
    lpdm_mole_fractions = lpdm_footprint_convolve(lpdm_footprint, wrf_fluxes)
    _LOGGER.info("Have LPDM mole fractions")
    lpdm_mole_fractions = lpdm_mole_fractions.load()
    _LOGGER.info("Loaded LPDM mole fractions")
    fig = compare_wrf_lpdm_mole_fractions_for_month(
        wrf_mole_fractions, lpdm_mole_fractions, args.year, args.month
    )
    save_nonsparse_netcdf(
        wrf_fluxes,
        os.path.join(
            args.output_dir,
            (
                "WRF_fluxes_{year:04d}_{month:02d}_"
                "{flux_interval:02d}hrly_{res:03d}km.nc4"
            ).format(
                year=args.year,
                month=args.month,
                flux_interval=FLUX_INTERVAL,
                res=FLUX_RESOLUTION,
            ),
        ),
    )
    _LOGGER.info("Saved WRF fluxes")
    save_nonsparse_netcdf(
        wrf_mole_fractions,
        os.path.join(
            args.output_dir,
            (
                "WRF_mole_fractions_{year:04d}_{month:02d}_"
                "{flux_interval:02d}hrly_{res:03d}km.nc4"
            ).format(
                year=args.year,
                month=args.month,
                flux_interval=FLUX_INTERVAL,
                res=FLUX_RESOLUTION,
            ),
        ),
    )
    _LOGGER.info("Saved WRF mole fractions")
    save_nonsparse_netcdf(
        lpdm_mole_fractions,
        os.path.join(
            args.output_dir,
            (
                "LPDM_mole_fractions_{year:04d}_{month:02d}_"
                "{flux_interval:02d}hrly_{res:03d}km.nc4"
            ).format(
                year=args.year,
                month=args.month,
                flux_interval=FLUX_INTERVAL,
                res=FLUX_RESOLUTION,
            ),
        ),
    )
    _LOGGER.info("Saved LPDM mole fractions")
    # save_sparse_influences(
    #     lpdm_footprint,
    #     os.path.join(
    #         args.output_dir,
    #         (
    #             "LPDM_{year:04d}_{month:02d}_{flux_interval:02d}hrly_{res:03d}km"
    #             "_flux_time_aligned_sparse_molar_footprints.nc4"
    #         ).format(
    #             year=args.year,
    #             month=args.month,
    #             flux_interval=FLUX_INTERVAL,
    #             res=FLUX_RESOLUTION,
    #         ),
    #     ),
    # )
    # _LOGGER.info("Saved aligned footprint (sparse)")
    # save_nonsparse_netcdf(
    #     lpdm_footprint,
    #     os.path.join(
    #         args.output_dir,
    #         (
    #             "LPDM_{year:04d}_{month:02d}_{flux_interval:02d}hrly_{res:03d}km"
    #             "_flux_time_aligned_molar_footprints.nc4"
    #         ).format(
    #             year=args.year,
    #             month=args.month,
    #             flux_interval=FLUX_INTERVAL,
    #             res=FLUX_RESOLUTION,
    #         ),
    #     ),
    # )
    # _LOGGER.info("Saved aligned footprint (dense)")
