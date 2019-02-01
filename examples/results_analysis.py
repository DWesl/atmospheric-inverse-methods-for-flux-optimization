#!/usr/bin/env python
# ~*~ coding: utf-8 ~*~
"""Plot results from inversion.

Only a single run.  Currently set up for plots from
month_inversion_magic_dask
"""
from __future__ import print_function, division
import datetime
import glob
import sys
import os

import numpy as np
import scipy.stats
import scipy.special
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeat
import iris
import dask.array as da
import xarray.plot
print(datetime.datetime.now(), "Imports")
sys.stdout.flush()

YEAR = 2010
MONTH = 7

NOISE_FUNCTION = "exp"
NOISE_LENGTH = 200
NOISE_TIME_FUN = "exp"
NOISE_TIME_LEN = 21
INV_FUNCTION = "exp"
INV_LENGTH = 200
INV_TIME_FUN = "exp"
INV_TIME_LEN = 21

FLUX_INTERVAL = 6
FLUX_RESOLUTION = 27


def write_console_message(msg):
    """Write a message to stdout and flush output streams."""
    sys.stderr.flush()
    print(datetime.datetime.now(), msg, flush=True)


PRIOR_PATH = (
    "../data_files/"
    "{year:04d}-{month:02d}_osse_bio_priors_{interval:d}h_{res:d}km_"
    "noise_{fun:s}{len:d}km_{time_fun:s}{time_len:d}d_exp3h.nc"
).format(
    year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
    fun=NOISE_FUNCTION, len=NOISE_LENGTH, time_fun=NOISE_TIME_FUN,
    time_len=NOISE_TIME_LEN)
# 2010-07_monthly_inversion_06h_027km_noiseexp100kmexp14d_icovexp100kmexp14d_output.nc4
POSTERIOR_PATH = (
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d"
    "_icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d"
    "_output.nc4"
).format(year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
         noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
         noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
         invfun=INV_FUNCTION, invlen=INV_LENGTH,
         inv_time_fun=INV_TIME_FUN, inv_time_len=INV_TIME_LEN)

WRF_CRS = ccrs.LambertConformal(
    standard_parallels=(30, 60), central_latitude=40,
    central_longitude=-96, false_easting=0, false_northing=0,
    globe=ccrs.Globe(semimajor_axis=6370e3, semiminor_axis=6360e3,
                     ellipse=None))
LPDM_PROJ = ccrs.LambertConformal(
    central_longitude=-96, central_latitude=40, standard_parallels=[30, 60],
    false_easting=3347998.5116325677, false_northing=2470499.376688077,
    globe=ccrs.Globe(semimajor_axis=6370e3, semiminor_axis=6370e3,
                     ellipse=None))

BIG_LAKES = cfeat.NaturalEarthFeature(
    "physical", "lakes", "110m",
    edgecolor="gray", facecolor="none", linewidth=.5)
STATES = cfeat.NaturalEarthFeature(
    "cultural", "admin_1_states_provinces_lines", "110m",
    edgecolor="gray", facecolor="none", linewidth=.5)

TRACER_NAMES = [
    "diurnal_bio",
    "fossil",
    "ocean",
    "biomass_burn",
    "biofuel",
    "ship",
    "posterior_bio",
    "boundaries",
    "prior_bio",
    "",
]

# WEST_BOUNDARY_LPDM = 2.7e6
# WEST_BOUNDARY_WRF = WRF_CRS.transform_point(
#     WEST_BOUNDARY_LPDM, 0, LPDM_PROJ)[0]

# Estimates for West Virginia, roughly
WEST_BOUNDARY_WRF = 1.13e6
EAST_BOUNDARY_WRF = 1.52e6
SOUTH_BOUNDARY_WRF = -1.76e5
NORTH_BOUNDARY_WRF = 2.02e5

LPDM_BOUNDS = LPDM_PROJ.transform_points(
    WRF_CRS,
    np.array([WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF]),
    np.array([SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF]),
)

print(datetime.datetime.now(), "Constants")
sys.stdout.flush()


def plot_realizations(data_array):
    """Plot a `Dataarray` with realizations."""
    time_dim = [dim for dim in data_array.dims if "time" in dim][0]
    x_dim = [dim for dim in data_array.dims if "_x" in dim][0]
    y_dim = [dim for dim in data_array.dims if "_y" in dim][0]
    xarray.plot.pcolormesh(
        data_array.isel(**{time_dim: slice(3, None, 96),
                           "realization": slice(3)}),
        x_dim, y_dim,
        col=time_dim, row="realization", cmap="RdBu_r", center=0,
        aspect=1.35, size=1.8,
        subplot_kws=dict(projection=WRF_CRS),
    )

    post_fig = plt.gcf()
    axes = post_fig.axes
    try:
        axes[-1].set_ylabel(
            "{long_name:s}  (${units:s}$)".format(**data_array.attrs))
    except KeyError:
        pass
    xlim = data_array[x_dim][[0, -1]]
    ylim = data_array[y_dim][[0, -1]]

    for ax in axes[:-1]:
        ax.coastlines()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_feature(cfeat.BORDERS)

    return post_fig


def plot_fluxes(data_array):
    """Plot a single flux realization."""
    time_dim = [dim for dim in data_array.dims if "time" in dim][0]
    x_dim = [dim for dim in data_array.dims if "_x" in dim][0]
    y_dim = [dim for dim in data_array.dims if "_y" in dim][0]
    xarray.plot.pcolormesh(
        data_array.isel(**{time_dim: slice(3, None, 32)}),
        x_dim, y_dim,
        col=time_dim, col_wrap=3, cmap="RdBu_r", center=0,
        aspect=1.35, size=1.8,
        subplot_kws=dict(projection=WRF_CRS),
    )

    post_fig = plt.gcf()
    axes = post_fig.axes
    try:
        axes[-1].set_ylabel(
            "{long_name:s} (${units:s}$)".format(**data_array.attrs))
    except KeyError:
        pass
    xlim = data_array[x_dim][[0, -1]]
    ylim = data_array[y_dim][[0, -1]]

    for ax in axes[:-1]:
        ax.coastlines()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_feature(cfeat.BORDERS)

    post_fig.subplots_adjest(top=.9, bottom=.0256, left=.0216,
                             right=.82, hspace=.2, wspace=.08)
    return post_fig


print(datetime.datetime.now(), "Functions")
sys.stdout.flush()

PRIOR_CUBES = iris.load(PRIOR_PATH)
PRIOR_DS = xarray.open_dataset(
    PRIOR_PATH, chunks=dict(realization=1, flux_time=8*7)).set_coords(
        "lambert_conformal_conic")  # .rename(
    # dict(south_north="dim_y", west_east="dim_x"))

PRIOR_CUBES.sort(key=lambda cube: cube.name())
PRIOR_CUBES.sort(key=lambda cube: len(cube.name()))

for name, var in PRIOR_DS.data_vars.items():
    num_end = name.rindex("_")
    if num_end < 5:
        num_end = None
    tracer_num = int(name[5:num_end])
    long_name = TRACER_NAMES[tracer_num - 1]
    if name.endswith("_noisy"):
        long_name += "_with_added_noise"
    var.attrs["long_name"] = long_name
    try:
        var.attrs["units"] = "μ" + var.attrs["units"]
        var *= 1e6
    except KeyError:
        print(name)
        pass


NOISE_STD_DS = xarray.open_dataset(
    "../data_files/{year:04d}_MsTMIP_flux_std.nc4".format(
        year=YEAR, month=MONTH),
    chunks=dict(Time=21*8)
)[["E_TRA1"]].rolling(
    center=True, min_periods=3, Time=8*21
).mean(dim="Time", keep_attrs=True).sel(
    Time=slice("2010-06-16", "2010-07-31")
).mean("Time", keep_attrs=True)
NOISE_STD_DS["E_TRA1"].attrs["units"] = "umol/m^2/s"

for name, var in NOISE_STD_DS.data_vars.items():
    if name.startswith("E_TRA"):
        tracer_num = int(name[5:])
    elif name.startswith("tracer_"):
        tracer_num = int(name[7:])
    else:
        continue
    long_name = TRACER_NAMES[tracer_num - 1]
    long_name += "_rms"
    var.attrs["long_name"] = long_name
    if var.attrs["units"] == "mol km^-2 hr^-1":
        var /= 3.6e3
        var.attrs["units"] = "μmol/m^2/s"

NOISE_STD_DS.coords["south_north"] = PRIOR_DS.coords["dim_y"].data
NOISE_STD_DS.coords["west_east"] = PRIOR_DS.coords["dim_x"].data

NOISE_STD_DS["E_TRA7"] = NOISE_STD_DS["E_TRA1"]

INVERSION_DS = xarray.open_dataset(POSTERIOR_PATH,
                                   chunks=dict(realization=1, flux_time=8*14))
PSEUDO_OBS_DS = INVERSION_DS["pseudo_observations"]
POSTERIOR_DS = INVERSION_DS[["posterior", "prior", "truth"]].rename(
    dict(dim_x="projection_x_coordinate", dim_y="projection_y_coordinate",
         flux_time="time"))
POSTERIOR_DS.coords["projection_x_coordinate"].attrs.update(
    dict(units="m", standard_name="projection_x_coordinate", axis="X"))
POSTERIOR_DS.coords["projection_y_coordinate"].attrs.update(
    dict(units="m", standard_name="projection_y_coordinate", axis="Y"))
POSTERIOR_DS.coords["time"] = POSTERIOR_DS.indexes["time"].round("S")

for var in POSTERIOR_DS.data_vars.values():
    var *= 1e6
    var.attrs["units"] = "μ" + var.attrs["units"]

SMALL_POSTERIOR_DS = POSTERIOR_DS.sel(
    projection_x_coordinate=slice(WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF),
    projection_y_coordinate=slice(SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF),
)
SMALL_PRIOR_DS = PRIOR_DS.sel(
    dim_x=slice(WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF),
    dim_y=slice(SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF))

PSEUDO_OBS_DS = xarray.open_mfdataset(
    "observation_realizations_for_06h_0?.nc4",
    concat_dim="observation",
    chunks=dict(observation=24*7),
).set_index(
    observation=("observation_time", "site")).transpose(
    "realization", "observation")

INFLUENCE_PATHS = ["/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "ACT-America_trial5/2010/01/GROUP1",
                   "/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "candidacy_more_towers/2010/01/GROUP1"]


COLLAPSED_INFLUENCE_DS = xarray.open_dataset(
    "../data_files/LPDM_2010_01_31day_027km_molar_footprints.nc4",
).set_coords(
    ["observation_time", "time_before_observation",
     "lpdm_configuration", "wrf_configuration"])
FULL_INFLUENCE_DS = xarray.open_mfdataset(
    [name
     for path in INFLUENCE_PATHS
     for name in glob.iglob(os.path.join(
         path,
         ("LPDM_2010_01*{flux_interval:02d}hrly_{res:03d}km_"
          "molar_footprints.nc4").format(
             flux_interval=FLUX_INTERVAL, res=FLUX_RESOLUTION)))],
    concat_dim="site",
    chunks=dict(time_before_observation=4 * 7, observation_time=24),
).set_coords(["lpdm_configuration", "wrf_configuration"])

print(datetime.datetime.now(), "Files", flush=True)
sys.stderr.flush()

############################################################
# Plot standard deviations
fig, axes = plt.subplots(
    1, 1, figsize=(5.5, 3.3), subplot_kw=dict(projection=WRF_CRS))
NOISE_STD_DS.E_TRA7.plot.pcolormesh(robust=True)
axes = fig.axes
axes[0].coastlines()
axes[1].set_ylabel("standard deviation of noise (μmol/m$^2$/s)")
fig.suptitle("Standard deviation of added noise")
axes[0].set_xlim(PRIOR_DS.coords["dim_x"][[0, -1]])
axes[0].set_ylim(PRIOR_DS.coords["dim_y"][[0, -1]])
axes[0].set_title("")
axes[0].add_feature(cfeat.BORDERS)
axes[0].add_feature(STATES)
axes[0].add_feature(BIG_LAKES)

fig.savefig("{year:04d}-{month:02d}_noise_standard_deviation.png".format(
    year=YEAR, month=MONTH))
plt.close(fig)
write_console_message("Made std plot")

############################################################
# Plot pseudo-observations
fig, axes = plt.subplots(len(PSEUDO_OBS_DS.site),
                         sharex=True,
                         figsize=(8, 1.5 * len(PSEUDO_OBS_DS.site)))
write_console_message("Made figure")
fig.autofmt_xdate()
pseudo_obs = PSEUDO_OBS_DS.pseudo_observations
for i, site in enumerate(set(PSEUDO_OBS_DS.site.values)):
    site_obs = pseudo_obs.sel(site=site)
    axes[i].plot(site_obs.observation_time.values, site_obs.values.T)
    axes[i].text(0.01, 0.98, site, transform=axes[i].transAxes,
                 horizontalalignment="left", verticalalignment="top")

fig.suptitle("Pseudo-observations used in inversion")
# fig.savefig("{year:04d}-{month:02d}_pseudo_obs_afternoon.png".format(
#     year=YEAR, month=MONTH))
fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_pseudo_obs_afternoon.pdf"
    .format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN))
plt.close(fig)
write_console_message("Made pseudo-obs plot")

############################################################
# Plot "truth", prior, and posterior side-by-side
for_plotting = xarray.concat((POSTERIOR_DS.prior.isel(realization=0),
                              POSTERIOR_DS.posterior.isel(realization=0)),
                             dim="type")
del for_plotting.coords["realization"]

# e_tra7_for_plot = PRIOR_DS["E_TRA7"].rename(
#     dict(dim_x="projection_x_coordinate", dim_y="projection_y_coordinate",
#          flux_time="time"))
for_plotting = xarray.concat((POSTERIOR_DS.truth, for_plotting), dim="type")
for_plotting.coords["type"] = ['"Truth"', "Prior", "Posterior"]
for_plotting.persist()

xlim = for_plotting.coords["projection_x_coordinate"][[0, -1]]
ylim = for_plotting.coords["projection_y_coordinate"][[0, -1]]

plots = for_plotting.isel(time=slice(71, None, 40)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    col="type", row="time", subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    center=0, vmin=-10, vmax=10,
    cmap="RdBu_r", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.cbar.ax.set_ylabel("CO$_2$ Flux (μmol/m$^2$/s)")
plots.axes[0, 0].set_title('"Truth"')
plots.axes[0, 1].set_title("Prior")
plots.axes[0, 2].set_title("Posterior")

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_osse_realization.png".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN),
    dpi=400)
plt.close(plots.fig)
write_console_message("Done realization plot")

############################################################
# Plot tower locations
fig, ax = plt.subplots(
    1, 1, subplot_kw=dict(projection=WRF_CRS), figsize=(4, 3))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.coastlines()
ax.add_feature(BIG_LAKES)
ax.add_feature(cfeat.BORDERS)
ax.add_feature(STATES)
ax.scatter(pseudo_obs.tower_lon, pseudo_obs.tower_lat,
           transform=WRF_CRS.as_geodetic())
fig.suptitle("WRF domain and tower locations")
fig.savefig("tower_locations.pdf")
plt.close(fig)
write_console_message("Done tower loc plot")

############################################################
# Plot differences
differences = (for_plotting.isel(type=slice(1, None)) -
               for_plotting.isel(type=0))
differences.persist()
plots = differences.isel(time=slice(68-1, None, 40)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    col="type", row="time", subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    center=0, vmin=-10, vmax=10,
    cmap="RdBu_r", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.cbar.ax.set_ylabel("CO$_2$ Flux (μmol/m$^2$/s)")
plots.axes[0, 0].set_title("Prior $-$ \"Truth\"")
plots.axes[0, 1].set_title("Posterior $-$ \"Truth\"")

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_osse_errors.png".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(plots.fig)
write_console_message("Done error plot")

############################################################
# Plot gain over time
gain = 1 - (da.fabs(differences.sel(type="Posterior")) /
            da.fabs(differences.sel(type="Prior")))
gain.attrs["long_name"] = "inversion_gain"
gain.attrs["units"] = "1"
# gain.load()

plots = gain.isel(time=slice(68-1, None, 20)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    row="time", col_wrap=3, subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    vmin=0, vmax=1, cmap="viridis", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_pointwise_gain.png"
    .format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(plots.fig)
write_console_message("Done pointwise gain")

############################################################
# Find and plot gains for all realizations
all_differences = xarray.concat(
    (POSTERIOR_DS.prior - for_plotting.sel(type='"Truth"'),
     POSTERIOR_DS.posterior - for_plotting.sel(type='"Truth"')),
    dim="type")
all_differences.coords["type"] = ["prior_error", "posterior_error"]

print(datetime.datetime.now(), "Getting January means east of line")
sys.stdout.flush()
time_mean_error = all_differences.mean("time")
time_mean_error.load()
all_mean_error = time_mean_error.mean(
    ("projection_x_coordinate", "projection_y_coordinate"))
all_mean_error.load()
small_mean_error = time_mean_error.sel(
    projection_x_coordinate=slice(WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF),
    projection_y_coordinate=slice(SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF)
).mean(
    ("projection_x_coordinate", "projection_y_coordinate"))
small_mean_error.load()
print(datetime.datetime.now(), "Have means")
sys.stdout.flush()

total_gain = 1 - (da.fabs(all_mean_error.sel(type="posterior_error")) /
                  da.fabs(all_mean_error.sel(type="prior_error")))
total_gain.load()

fig = plt.figure()
total_gain.plot.hist(range=(-2, 1), bins=15)
fig.suptitle("Total gain")
plt.xlabel("Gain in total flux")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_total_gain_hist.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done large gain plot")

small_gain = 1 - (da.fabs(small_mean_error.sel(type="posterior_error")) /
                  da.fabs(small_mean_error.sel(type="prior_error")))
small_gain.load()

fig = plt.figure()
small_gain.plot.hist(range=(-2, 1), bins=15)
fig.suptitle("Total gain on small domain")
plt.xlabel("Gain in flux in small domain")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_small_gain_hist.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done small gain plot")

############################################################
# Describe the distribution
with open(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_gain_dist.txt".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN), "w") as result_file:
    print(datetime.datetime.now(), "Total gain", file=result_file)
    print(scipy.stats.describe(total_gain), file=result_file)

    print(total_gain.quantile((0, .2, .25, .4, .5, .6, .75, .8, 1)),
          file=result_file)

with open(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_small_gain_dist.txt".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN), "w") as result_file:
    print(datetime.datetime.now(), "Gain in small domain", file=result_file)
    print(scipy.stats.describe(small_gain), file=result_file)

    print(small_gain.quantile((0, .2, .25, .4, .5, .6, .75, .8, 1)),
          file=result_file)
write_console_message("Done gain statistics files")

############################################################
# Plot timeseries of mean flux
spatial_avg_differences = all_differences.mean(
    ["projection_x_coordinate", "projection_y_coordinate"])

fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
spatial_avg_differences.sel(
    type="prior_error", realization=0).plot.line(
    "-", label="Prior")
spatial_avg_differences.sel(
    type="posterior_error", realization=0).plot.line(
    "--", label="Posterior")

ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)
ax.axhline(0, color="black", linewidth=.75)

plt.legend()
ax.set_ylabel("Average flux error over whole domain\n(μmol/m²/s)")
ax.set_xlabel("")
plt.title("Spatial average flux error")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_spatial_"
    "avg_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done timeseries")


small_spatial_avg_differences = all_differences.sel(
    projection_x_coordinate=slice(WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF),
    projection_y_coordinate=slice(SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF)
).mean(["projection_x_coordinate", "projection_y_coordinate"])

fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
small_spatial_avg_differences.sel(
    type="prior_error", realization=0).plot.line(
    "-", label="Prior")
small_spatial_avg_differences.sel(
    type="posterior_error", realization=0).plot.line(
    "--", label="Posterior")

ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)

ax.axhline(0, color="black", linewidth=.75)

plt.legend()
ax.set_ylabel("Average flux error over West Virginia\n(μmol/m²/s)")
ax.set_xlabel("")
plt.title("Spatial average flux error")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_small_spatial_"
    "avg_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done small timeseries")


############################################################
# Plot timeseries of increment
spatial_avg_increment = (
    spatial_avg_differences.sel(type="posterior_error") -
    spatial_avg_differences.sel(type="prior_error")
)
fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
spatial_avg_increment.isel(realization=0).plot.line('-')

# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)

ax.axhline(0, color="black", linewidth=.75)
ax.set_ylabel("Average increment over whole domain\n(μmol/m²/s)")
ax.set_xlabel("")
plt.title("Spatial average increment")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_spatial_"
    "avg_increment_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)


small_spatial_avg_increment = (
    small_spatial_avg_differences.sel(type="posterior_error") -
    small_spatial_avg_differences.sel(type="prior_error")
)
fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
try:
    ax.set_xlim(mpl.dates.datestr2num(
        ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
    small_spatial_avg_increment.isel(realization=0).plot.line('-')
except ValueError:
    ax.set_xlim(mpl.dates.datestr2num(
        ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
    small_spatial_avg_increment.isel(realization=0).plot.line('-')

# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)

ax.axhline(0, color="black", linewidth=.75)
ax.set_ylabel("Average increment over West Virginia\n(μmol/m²/s)")
ax.set_xlabel("")
plt.title("Spatial average increment")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_small_spatial_"
    "avg_increment_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done increment plots")

############################################################
# Plot histogram of average flux errors
error_range = da.asarray(
    [all_mean_error.min(), all_mean_error.max()]).compute()
fig, ax = plt.subplots(1, 1)
# ax.hist([all_mean_error.sel(type="prior_error"),
#              all_mean_error.sel(type="posterior_error")],
#             range=[-.1, .1],
#             label=["Prior", "Posterior"])
all_mean_error.to_dataframe(
    name="error"
).loc[:, "error"].unstack(0).loc[
    :, ["prior_error", "posterior_error"]
].plot.kde(
    ax=ax, subplots=False, xlim=(-.1, .1), ylim=(0, None),
)
# sns.kdeplot(all_mean_error.sel(type="prior_error"), ax=ax,
#             shade=True, label="Prior")
# sns.kdeplot(all_mean_error.sel(type="posterior_error"), ax=ax,
#             shade=True, label="Posterior")
ax.legend(["Prior Means", "Posterior Means"])
ax.set_ylabel("Density")
ax.set_xlabel("Error in mean estimate (μmol/m$^2$/s)")


fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_flux_error_density.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))

plt.close(fig)

small_error_range = da.asarray(
    [small_mean_error.min(), small_mean_error.max()]).compute()
fig, ax = plt.subplots(1, 1)
# ax.hist([small_mean_error.sel(type="prior_error"),
#          small_mean_error.sel(type="posterior_error")],
#         range=[-4, 4],
#         label=["Prior", "Posterior"])
ax.set_xlabel("West Virginia mean flux error (μmol/m$^2$/s)")
small_mean_error.to_dataframe(
    name="error"
).loc[:, "error"].unstack(0).loc[
    :, ["prior_error", "posterior_error"]
].plot.kde(
    ax=ax, subplots=False, xlim=(-4, 4), ylim=(0, None),
)
ax.legend(["Prior Means", "Posterior Means"])
ax.set_ylabel("Density")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_small_flux_error_density.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Wrote error densities")

############################################################
# Gain vs. magnitude of prior error
fig = plt.figure(figsize=(4, 2.5))
plt.subplots_adjust(left=.24, bottom=.2)
plt.plot(total_gain, all_mean_error.sel(type="prior_error"), ".")
plt.xlabel("Gain")
plt.ylabel("Prior error (μmol/m$^2$/s)")
plt.xlim(-1, 1)

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_gain_prior_mag.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

fig = plt.figure(figsize=(4, 2.5))
plt.subplots_adjust(left=.24, bottom=.2)
plt.plot(small_gain, small_mean_error.sel(type="prior_error"), ".")
plt.xlabel("Gain")
plt.ylabel("Prior error (μmol/m$^2$/s)")
plt.xlim(-1, 1)

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_small_gain_prior_mag.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

############################################################
# Error reduction
time_mean_error_var = time_mean_error.var("realization")

time_mean_error_reduction = 1 - da.sqrt(
    time_mean_error_var.sel(type="posterior_error") /
    time_mean_error_var.sel(type="prior_error"))

fig, ax = plt.subplots(
    1, 1, subplot_kw=dict(projection=WRF_CRS), figsize=(4, 2.4))
time_mean_error_reduction.plot.pcolormesh(
    vmin=-.5, vmax=.5, center=0, cmap="RdBu_r")

ax.set_title("Error reduction")
ax.coastlines()
ax.set_xlim(xlim)
ax.set_ylim(ylim)

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_error_reduction.png".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)


############################################################
# Plot tower locations and influence functions

fig, axes = plt.subplots(
    1, 1, figsize=(5, 3), subplot_kw=dict(projection=LPDM_PROJ))

COLLAPSED_INFLUENCE_DS.H.plot(robust=True)
plt.scatter(FULL_INFLUENCE_DS.coords["site_lons"],
            FULL_INFLUENCE_DS.coords["site_lats"],
            transform=LPDM_PROJ.as_geodetic(), color="red")

plt.title("Integrated influence functions")
fig.axes[-1].set_ylabel("H (ppmv/(mol/m$^2$/s))")
axes.set_xlim(COLLAPSED_INFLUENCE_DS.coords["dim_x"][[0, -1]])
axes.set_ylim(COLLAPSED_INFLUENCE_DS.coords["dim_y"][[0, -1]])

axes.coastlines()
axes.add_feature(cfeat.BORDERS)
axes.add_feature(BIG_LAKES, facecolor="none")
axes.add_feature(STATES)
# axes.axvline(WEST_BOUNDARY_LPDM, color="white")

fig.savefig("{year:04d}-{month:02d}_integrated_influence_functions.png".format(
    year=YEAR, month=MONTH), dpi=400)
plt.close(fig)
write_console_message("Done all figures")
