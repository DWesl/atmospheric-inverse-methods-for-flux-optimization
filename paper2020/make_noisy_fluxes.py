#!/usr/bin/env python
"""Run an identical twin flux inversion OSSE with real data.

Use xarray/dask to grab influence functions and priors from netCDF
files.
"""
from __future__ import print_function, division, unicode_literals
# Four runs in 22min, 2h cpu time, 54 GiB memory for one tracer.

import datetime
import os.path
import glob
import sys

import pandas as pd
import dateutil.tz
import numpy as np
import cf_units
import netCDF4
import xarray
import wrf

try:
    THIS_DIR = os.path.dirname(__file__)
except NameError:
    THIS_DIR = os.getcwd()

sys.path.insert(0, os.path.join(
    THIS_DIR, "..", "src"))
sys.path.append(THIS_DIR)

import atmos_flux_inversion.correlations
import atmos_flux_inversion.covariances
from atmos_flux_inversion.linalg import kron
from atmos_flux_inversion.util import kronecker_product
from atmos_flux_inversion.covariances import CorrelationStandardDeviation
from atmos_flux_inversion.noise import gaussian_noise
import cf_acdd

TRUE_FLUXES_DIR = os.path.join(THIS_DIR, "..", "data_dir")

# dask.set_options(pool=multiprocessing.pool.ThreadPool(16))
FLUX_INTERVAL = 6
"""The interval at which fluxes become available in hours.

Fluxes are usually integrated forward from hourly input, but I can't
solve for that in a reasonable timeframe.

This determines both the input and the output time resolution as well
as what the inversion solves for.

Note
----
Must divide twenty-four.
"""
FLUX_RESOLUTION = 27

FLUX_FILES = glob.glob(os.path.join(
    TRUE_FLUXES_DIR,
    "2010-07_wrf_fluxes_all_{interval:02d}hrly_{res:d}km.nc".format(
        interval=FLUX_INTERVAL, res=FLUX_RESOLUTION)))
FLUX_FILES.sort()

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
UTC = dateutil.tz.tzutc()
SECONDS_PER_HOUR = 3600

CO2_MOLAR_MASS = 16 * 2 + 12.01
"""Molar mass of CO2 (g/mol).

Used to convert WRF fluxes to units expected by observation operator.
"""
CO2_MOLAR_MASS_UNITS = cf_units.Unit("g/mol")
FLUX_UNITS = cf_units.Unit("mol/m^2/s")

FLUX_CHUNKS = HOURS_PER_DAY * 31 // FLUX_INTERVAL
"""How many flux times to treat at once.

Must be a multiple of day length.
"""
N_REALIZATIONS = 80
"""Number of realizations of the gaussian noise process to include.

This is used for calculating both the prior and observations fed to
the inversion code.
"""
CORRELATION_LENGTH = 84
GRID_RESOLUTION = FLUX_RESOLUTION
SPATIAL_CORRELATION_FUNCTION = (
    atmos_flux_inversion.correlations.ExponentialCorrelation
)
SP_CORR_STR = "exp"
import argparse
parser = argparse.ArgumentParser(description="Generate some noise")
parser.add_argument("corr_len", type=float, default=84,
                    help="correlation length (km)")
parser.add_argument("corr_fun", choices=["exp", "balg", "matn", "gaus"],
                    help="Spatial correlation function")
parser.add_argument("--realizations", type=int, default=N_REALIZATIONS,
                    help="Number of noise realizations")
args = parser.parse_args()
CORRELATION_LENGTH = args.corr_len
N_REALIZATIONS = args.realizations
SP_CORR_STR = args.corr_fun

if SP_CORR_STR == "exp":
    SPATIAL_CORRELATION_FUNCTION = (
        atmos_flux_inversion.correlations.ExponentialCorrelation
    )
elif SP_CORR_STR == "balg":
    SPATIAL_CORRELATION_FUNCTION = (
        atmos_flux_inversion.correlations.BalgovindCorrelation
    )
elif SP_CORR_STR == "matn":
    SPATIAL_CORRELATION_FUNCTION = (
        atmos_flux_inversion.correlations.MaternCorrelation
    )
elif SP_CORR_STR == "gaus":
    SPATIAL_CORRELATION_FUNCTION = (
        atmos_flux_inversion.correlations.GaussianCorrelation
    )

with netCDF4.Dataset(FLUX_FILES[0]) as ds:
    WRF_PROJECTION = wrf.util.getproj(**wrf.util.get_proj_params(ds))
    WRF_CRS = WRF_PROJECTION.cartopy()

    NX = len(ds.dimensions["projection_x_coordinate"])
    NY = len(ds.dimensions["projection_y_coordinate"])

del ds

N_GRID_POINTS = NY * NX


def flush_output_streams():
    """Flush stdout and stderr."""
    sys.stdout.flush()
    sys.stderr.flush()


print(datetime.datetime.now(UTC).strftime("%c"),
      "Have constants, getting priors")
############################################################
# Read prior fluxes
FLUX_DATASET = xarray.open_mfdataset(
    FLUX_FILES,
    concat_dim="XTIME",
)
print(datetime.datetime.now(UTC).strftime("%c"), "Have fluxes, normalizing")
flush_output_streams()

# Many of the times are off by about four milliseconds.
# This difference is irrelevant here.
wrf_times = FLUX_DATASET["XTIME"].to_index().round("S")
timestamps = list(wrf_times)
timestamps[-1] += datetime.timedelta(hours=FLUX_INTERVAL / 4)
# timestamps[0] -= datetime.timedelta(hours=1)
wrf_new_times = pd.DatetimeIndex(timestamps,
                                 name="XTIME")
FLUX_DATASET.coords["Time"] = wrf_new_times

FLUX_DATASET = FLUX_DATASET.set_index(XTIME="Time")
FLUX_DATASET = FLUX_DATASET.rename(
    dict(
        XTIME="Time", projection_y_coordinate="south_north",
        projection_x_coordinate="west_east"
    )
)

# Select out only full days so we have something the covariances can
# deal with.
FLUX_DATASET = FLUX_DATASET.sel(
    Time=slice("2010-06-04", "2010-08-01")).isel(Time=slice(None, -1))

WRF_DX = FLUX_DATASET.attrs["DX"]

assert WRF_DX / 1000 == FLUX_RESOLUTION

TRUE_FLUXES = FLUX_DATASET.get(["E_TRA{:d}".format(i + 1)
                                for i in range(10)]).isel(emissions_zdim=0)
TRUE_FLUXES_MATCHED = TRUE_FLUXES.rename(dict(
    south_north="dim_y", west_east="dim_x", Time="flux_time"))
for flux_part, flux_orig in zip(TRUE_FLUXES_MATCHED.data_vars.values(),
                                TRUE_FLUXES.data_vars.values()):
    unit = (cf_units.Unit(flux_orig.attrs["units"]))
    flux_part *= unit.convert(1, FLUX_UNITS)
    flux_part.attrs["units"] = str(FLUX_UNITS)

############################################################
# Define correlation constants and get covariances
print(datetime.datetime.now(UTC).strftime("%c"), "Getting covariances")
flush_output_streams()

DAILY_FLUX_TIMESCALE = 21
HOURLY_FLUX_TIMESCALE = 3
INTERVALS_PER_DAY = HOURS_PER_DAY // FLUX_INTERVAL

spatial_correlations = (
    atmos_flux_inversion.correlations.HomogeneousIsotropicCorrelation.
    # First guess at correlation length on the order of previous studies
    from_function(
        SPATIAL_CORRELATION_FUNCTION(
            CORRELATION_LENGTH / GRID_RESOLUTION),
        (len(TRUE_FLUXES_MATCHED.coords["dim_y"]),
         len(TRUE_FLUXES_MATCHED.coords["dim_x"])),
        is_cyclic=False,
    )
)
print(datetime.datetime.now(UTC).strftime("%c"), "Have spatial correlations")
flush_output_streams()
# Look into prescribing negative correlations between day and night
hour_correlations = (
    atmos_flux_inversion.correlations.HomogeneousIsotropicCorrelation.
    from_function(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            HOURLY_FLUX_TIMESCALE / FLUX_INTERVAL),
        (INTERVALS_PER_DAY,),
        is_cyclic=True
    )
)
hour_correlations_matrix = hour_correlations.dot(np.eye(
    hour_correlations.shape[0]))
print(datetime.datetime.now(UTC).strftime("%c"), "Have hourly correlations")
flush_output_streams()

day_correlations = (
    atmos_flux_inversion.correlations.make_matrix(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            DAILY_FLUX_TIMESCALE
        ),
        (len(TRUE_FLUXES_MATCHED.coords["flux_time"]) *
         FLUX_INTERVAL // HOURS_PER_DAY,)))
print(datetime.datetime.now(UTC).strftime("%c"), "Have daily correlations")
flush_output_streams()
temporal_correlations = kron(day_correlations,
                             hour_correlations_matrix)
print("Temporal:", type(temporal_correlations))
print(datetime.datetime.now(UTC).strftime("%c"), "Have temporal correlations")
flush_output_streams()

# I would like to add a fixed minimum at some point.
# full stds would then be sqrt(fixed^2 + varying^2)
# average seasonal variation (or some fraction thereof) might work.
# x2 for MsTMIP fluxes not getting full uncertainty
# x5 for MsTMIP representing monthly fluxes, where I'm working with sub-daily
FLUX_VARIANCE_VARYING_FRACTION = 2. * 5.
flux_std_pattern = xarray.open_dataset(
    "../data_dir/2010_MsTMIP_flux_std.nc4", chunks=dict(Time=8 * 21)).get(
    ["E_TRA{:d}".format(i + 1) for i in range(1)])  # .isel(emissions_zdim=0)


# Ensure units work out
for flux_part in flux_std_pattern.data_vars.values():
    unit = cf_units.Unit(flux_part.attrs["units"])
    flux_part *= (
        unit
    ).convert(1, FLUX_UNITS)
    flux_part.attrs["units"] = str(FLUX_UNITS)

osse_prior_dataset = TRUE_FLUXES_MATCHED.copy()
flux_stds = FLUX_VARIANCE_VARYING_FRACTION * flux_std_pattern["E_TRA1"].sel(
    Time=TRUE_FLUXES_MATCHED.indexes["flux_time"]
).mean("Time")
spatial_covariance = CorrelationStandardDeviation(
    spatial_correlations, flux_stds.data
)

# Save to ensure I have only one realization
for flux_name, flux_vals in TRUE_FLUXES_MATCHED.data_vars.items():
    if not any(flux_name.endswith(char) for char in "167"):
        continue

    prior_covariance = kronecker_product(
        temporal_correlations,
        spatial_covariance
    )
    print("Covariance:", type(prior_covariance))
    print(datetime.datetime.now(UTC).strftime("%c"), "Have covariances")
    flush_output_streams()

    prior_var_atts = flux_vals.attrs.copy()
    prior_var_atts.update(dict(
        long_name="{name:s}_prior_fluxes".format(name=flux_name),
        units=flux_vals.attrs["units"],
        description=("prior fluxes for a monthlong "
                     "identical-twin OSSE inversion study"),
        origin="osse_noisy_fluxes.py"))

    prior_flux_vals = (
        flux_vals.data[np.newaxis, :, :, :] +
        gaussian_noise(prior_covariance, N_REALIZATIONS).reshape(
            (N_REALIZATIONS,) + flux_vals.shape))
    print(datetime.datetime.now(UTC).strftime("%c"),
          "Have noisy fluxes; adding to dataset")
    flush_output_streams()
    osse_prior_dataset[flux_name + "_noisy"] = xarray.DataArray(
        # Hopefully this will let dask finish its job
        data=prior_flux_vals.persist(),
        coords=flux_vals.coords,
        dims=("realization",) + flux_vals.dims,
        name="prior",
        attrs=prior_var_atts
    ).transpose(
        "realization", "flux_time", "dim_y", "dim_x")
    flux_stds.attrs["cell_methods"] = "standard_deviation: MsTMIP_realization"
    osse_prior_dataset[flux_name + "_stds"] = flux_stds.rename(
        realization="MsTMIP_realization")

osse_prior_dataset.coords["realization"] = np.arange(
    N_REALIZATIONS, dtype=np.int8)
osse_prior_dataset.coords["realization"].attrs.update(dict(
    standard_name="realization",
    long_name="realization_of_the_noise_process"))
osse_prior_dataset.attrs.update(cf_acdd.global_attributes_dict())
print(datetime.datetime.now(UTC).strftime("%c"),
      "Have all prior noise, chunking")
flush_output_streams()

encoding = {name: {"_FillValue": -1e38, "zlib": True}
            for name in osse_prior_dataset.data_vars}
encoding.update({name: {"_FillValue": None, "zlib": True}
                 for name in osse_prior_dataset.coords})
# osse_prior_dataset = osse_prior_dataset.chunk(
#     dict(flux_time=FLUX_CHUNKS, dim_y=NY, dim_x=NX,
#          realization=N_REALIZATIONS))
print(datetime.datetime.now(UTC).strftime("%c"),
      "Noisy fluxes chunked, saving")
flush_output_streams()
osse_prior_dataset.to_netcdf(
    "../data_dir/2010-07_osse_bio_priors_{interval:d}h_{res:d}km"
    "_noise_{sp_fun:s}{length:g}km_exp{day_time:d}d_exp{hour_time:d}h.nc"
    .format(
        interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        length=CORRELATION_LENGTH, sp_fun=SP_CORR_STR,
        day_time=DAILY_FLUX_TIMESCALE, hour_time=HOURLY_FLUX_TIMESCALE),
    encoding=encoding)
print(datetime.datetime.now(UTC).strftime("%c"), "Noisy fluxes saved, done")
flush_output_streams()
