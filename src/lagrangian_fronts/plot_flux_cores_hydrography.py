"""
plot_flux_cores_hydrography.py


Plot ONE ARGO-derived hydrographic field as a background and overlay:

    - ARGO Lagrangian flux cores
    - drifter Lagrangian flux cores
    - SEANOE Eulerian ACC fronts:
        NB, SAF, PF, SACCF, SB

Input formats are tailored to the actual analysis outputs:

sampled_map.nc
--------------
Regular (lat, lon) grid containing fields such as:

    temp_mean
    temp_std
    temp_smoothed_mean
    temp_zonal_gradient
    temp_meridional_gradient
    temp_gradient_magnitude

    psal_mean
    psal_std
    psal_smoothed_mean
    psal_zonal_gradient
    psal_meridional_gradient
    psal_gradient_magnitude


branch_cores.parquet
--------------------
Expected columns include:

    cell_id
    start_lon_bin
    start_lat_bin
    lon
    lat
    N_out_move
    U_out_all_magnitude_km_day
    theta_mu_out
    R1_out
    R2_out
    ridge_type
    missing_side
    component_id
    left_side_observable
    right_side_observable


SEANOE fronts
-------------
Expected variables:

    LonNB,    LatNB
    LonSAF,   LatSAF
    LonPF,    LatPF
    LonSACCF, LatSACCF
    LonSB,    LatSB


The script is purely for comparison/visualization.
It does not modify, connect, smooth, or reinterpret the Lagrangian cores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as pe

from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# DEFAULT PLOT SETTINGS
# ============================================================

DEFAULT_LAT_MIN = -80.0
DEFAULT_LAT_MAX = -30.0

DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (11, 11)

ARGO_COLOR = "deepskyblue"
DRIFTER_COLOR = "magenta"

ARGO_MARKER = "o"
DRIFTER_MARKER = "^"

DEFAULT_CORE_SIZE = 18.0

FRONT_COLOR = "black"
FRONT_WIDTH = 1.2
FRONT_HALO_WIDTH = 2.5


# ============================================================
# FIELD-SPECIFIC DEFAULTS
# ============================================================

FIELD_SETTINGS = {

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    "temp_mean": {
        "cmap": "turbo",
        "label": "Mean temperature [°C]",
        "percentiles": (1.0, 99.0),
        "symmetric": False,
        "zero_min": False,
    },

    "temp_smoothed_mean": {
        "cmap": "turbo",
        "label": "Smoothed mean temperature [°C]",
        "percentiles": (1.0, 99.0),
        "symmetric": False,
        "zero_min": False,
    },

    "temp_std": {
        "cmap": "viridis",
        "label": "Temperature standard deviation [°C]",
        "percentiles": (0.0, 99.0),
        "symmetric": False,
        "zero_min": True,
    },

    "temp_zonal_gradient": {
        "cmap": "RdBu_r",
        "label": "Zonal temperature gradient [°C km$^{-1}$]",
        "percentiles": (1.0, 99.0),
        "symmetric": True,
        "zero_min": False,
    },

    "temp_meridional_gradient": {
        "cmap": "RdBu_r",
        "label": "Meridional temperature gradient [°C km$^{-1}$]",
        "percentiles": (1.0, 99.0),
        "symmetric": True,
        "zero_min": False,
    },

    "temp_gradient_magnitude": {
        "cmap": "magma",
        "label": "Temperature gradient magnitude [°C km$^{-1}$]",
        "percentiles": (0.0, 95.0),
        "symmetric": False,
        "zero_min": True,
    },

    # --------------------------------------------------------
    # Salinity
    # --------------------------------------------------------

    "psal_mean": {
        "cmap": "turbo",
        "label": "Mean practical salinity [psu]",
        "percentiles": (5.0, 95.0),
        "symmetric": False,
        "zero_min": False,
    },

    "psal_smoothed_mean": {
        "cmap": "turbo",
        "label": "Smoothed mean practical salinity [psu]",
        "percentiles": (1.0, 99.0),
        "symmetric": False,
        "zero_min": False,
    },

    "psal_std": {
        "cmap": "viridis",
        "label": "Salinity standard deviation [psu]",
        "percentiles": (0.0, 99.0),
        "symmetric": False,
        "zero_min": True,
    },

    "psal_zonal_gradient": {
        "cmap": "RdBu_r",
        "label": "Zonal salinity gradient [psu km$^{-1}$]",
        "percentiles": (1.0, 99.0),
        "symmetric": True,
        "zero_min": False,
    },

    "psal_meridional_gradient": {
        "cmap": "RdBu_r",
        "label": "Meridional salinity gradient [psu km$^{-1}$]",
        "percentiles": (1.0, 99.0),
        "symmetric": True,
        "zero_min": False,
    },

    "psal_gradient_magnitude": {
        "cmap": "magma",
        "label": "Salinity gradient magnitude [psu km$^{-1}$]",
        "percentiles": (0.0, 97.0),
        "symmetric": False,
        "zero_min": True,
    },
}


# ============================================================
# SEANOE FRONT DEFINITIONS
# ============================================================

FRONT_VARIABLES = {
    "NB": ("LonNB", "LatNB"),
    "SAF": ("LonSAF", "LatSAF"),
    "PF": ("LonPF", "LatPF"),
    "SACCF": ("LonSACCF", "LatSACCF"),
    "SB": ("LonSB", "LatSB"),
}


# Different line patterns, same neutral color.
# This keeps the hydrographic raster and the two Lagrangian
# products as the main visual information.

FRONT_LINESTYLES = {
    "NB": ":",
    "SAF": "-",
    "PF": "--",
    "SACCF": "-.",
    "SB": (0, (6, 2, 1, 2, 1, 2)),
}


# ============================================================
# INPUT
# ============================================================

def load_background(
    path: Path,
    field: str,
) -> xr.DataArray:
    """
    Load one hydrographic field from sampled_map.nc.

    The supplied file has dimensions:
        lat = 50
        lon = 360

    and all hydrographic maps are stored directly on (lat, lon).
    """

    ds = xr.open_dataset(path)

    if field not in ds.data_vars:
        available = "\n    ".join(ds.data_vars)

        raise ValueError(
            f"Field '{field}' not found in {path}\n\n"
            f"Available fields:\n    {available}"
        )

    da = ds[field]

    if set(da.dims) != {"lat", "lon"}:
        raise ValueError(
            f"Expected '{field}' to have dimensions "
            f"('lat', 'lon'), found {da.dims}"
        )

    return da.transpose("lat", "lon")


def load_cores(
    path: Path,
) -> pd.DataFrame:
    """
    Load Lagrangian flux core cells.

    Coordinates are already supplied directly as:
        lon
        lat

    Therefore no reconstruction from grid-bin indices is needed.
    """

    df = pd.read_parquet(path)

    required = {
        "lon",
        "lat",
        "component_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}:\n"
            f"{sorted(missing)}"
        )

    df = df.copy()

    df = df[
        np.isfinite(df["lon"])
        & np.isfinite(df["lat"])
    ].copy()

    # Normalize longitude for plotting.
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0

    return df


def load_seanoe_fronts(
    path: Path,
    selected_fronts: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Read the requested SEANOE ACC fronts.

    Actual variables in the supplied NetCDF are:

        LonNB      LatNB
        LonSAF     LatSAF
        LonPF      LatPF
        LonSACCF   LatSACCF
        LonSB      LatSB
    """

    ds = xr.open_dataset(path)

    fronts = {}

    for name in selected_fronts:

        lon_name, lat_name = FRONT_VARIABLES[name]

        if lon_name not in ds or lat_name not in ds:
            raise ValueError(
                f"Could not find {lon_name}/{lat_name} "
                f"in {path}"
            )

        lon = np.asarray(ds[lon_name].values, dtype=float)
        lat = np.asarray(ds[lat_name].values, dtype=float)

        valid = np.isfinite(lon) & np.isfinite(lat)

        lon = lon[valid]
        lat = lat[valid]

        lon = ((lon + 180.0) % 360.0) - 180.0

        fronts[name] = (lon, lat)

    return fronts


# ============================================================
# FRONT GEOMETRY
# ============================================================

def split_at_dateline(
    lon: np.ndarray,
    lat: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split front paths wherever wrapped longitude jumps by >180°.

    Prevents matplotlib from connecting +180° directly to -180°
    across the interior of the map.
    """

    if len(lon) < 2:
        return []

    jumps = np.where(
        np.abs(np.diff(lon)) > 180.0
    )[0]

    starts = np.r_[0, jumps + 1]
    ends = np.r_[jumps + 1, len(lon)]

    segments = []

    for start, end in zip(starts, ends):

        if end - start >= 2:
            segments.append(
                (
                    lon[start:end],
                    lat[start:end],
                )
            )

    return segments


# ============================================================
# COLOR NORMALIZATION
# ============================================================

def choose_color_limits(
    values: np.ndarray,
    field: str,
    vmin: float | None,
    vmax: float | None,
    percentile_min: float | None,
    percentile_max: float | None,
) -> tuple[float, float]:
    """
    Resolve background color limits.

    Explicit --vmin / --vmax take priority.

    Otherwise use robust percentile limits from FIELD_SETTINGS.
    """

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError(
            f"No finite values found for field '{field}'."
        )

    settings = FIELD_SETTINGS.get(
        field,
        {
            "percentiles": (1.0, 99.0),
            "symmetric": False,
            "zero_min": False,
        },
    )

    default_pmin, default_pmax = settings["percentiles"]

    pmin = (
        default_pmin
        if percentile_min is None
        else percentile_min
    )

    pmax = (
        default_pmax
        if percentile_max is None
        else percentile_max
    )

    # --------------------------------------------------------
    # Signed gradients: symmetric around zero
    # --------------------------------------------------------

    if settings.get("symmetric", False):

        if vmax is None:
            vmax_auto = np.nanpercentile(
                np.abs(finite),
                pmax,
            )
        else:
            vmax_auto = abs(vmax)

        if vmin is None:
            vmin_auto = -vmax_auto
        else:
            vmin_auto = vmin

        return vmin_auto, vmax_auto

    # --------------------------------------------------------
    # Positive quantities
    # --------------------------------------------------------

    if vmin is None:

        if settings.get("zero_min", False):
            vmin_auto = 0.0
        else:
            vmin_auto = np.nanpercentile(
                finite,
                pmin,
            )

    else:
        vmin_auto = vmin

    if vmax is None:
        vmax_auto = np.nanpercentile(
            finite,
            pmax,
        )
    else:
        vmax_auto = vmax

    return vmin_auto, vmax_auto


# ============================================================
# MAP BOUNDARY
# ============================================================

def set_circular_boundary(ax) -> None:
    """
    Make the South Polar Stereo axes circular.
    """

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        200,
    )

    center = np.array([0.5, 0.5])
    radius = 0.5

    vertices = np.vstack(
        [
            np.sin(theta),
            np.cos(theta),
        ]
    ).T

    circle = mpath.Path(
        vertices * radius + center
    )

    ax.set_boundary(
        circle,
        transform=ax.transAxes,
    )


# ============================================================
# PLOT
# ============================================================

def plot_map(args) -> None:

    # --------------------------------------------------------
    # Load inputs
    # --------------------------------------------------------

    background = load_background(
        args.background,
        args.field,
    )

    argo = load_cores(
        args.argo_cores,
    )

    drifter = load_cores(
        args.drifter_cores,
    )

    fronts = load_seanoe_fronts(
        args.seanoe_fronts,
        args.fronts,
    )

    print(
        f"Background field: {args.field}"
    )

    print(
        f"ARGO core cells: {len(argo):,}"
    )

    print(
        f"Drifter core cells: {len(drifter):,}"
    )

    print(
        "SEANOE fronts: "
        + ", ".join(args.fronts)
    )

    # --------------------------------------------------------
    # Field style
    # --------------------------------------------------------

    settings = FIELD_SETTINGS.get(
        args.field,
        {
            "cmap": "viridis",
            "label": args.field,
            "percentiles": (1.0, 99.0),
            "symmetric": False,
            "zero_min": False,
        },
    )

    cmap = (
        args.cmap
        if args.cmap is not None
        else settings["cmap"]
    )

    colorbar_label = settings["label"]

    values = background.values

    vmin, vmax = choose_color_limits(
        values=values,
        field=args.field,
        vmin=args.vmin,
        vmax=args.vmax,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
    )

    print(
        f"Color scale: {vmin:.6g} to {vmax:.6g}"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    projection = ccrs.SouthPolarStereo(
        central_longitude=args.central_longitude
    )

    fig = plt.figure(
        figsize=(args.figsize, args.figsize)
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=projection,
    )

    ax.set_extent(
        [
            -180.0,
            180.0,
            args.lat_min,
            args.lat_max,
        ],
        crs=ccrs.PlateCarree(),
    )

    set_circular_boundary(ax)

    # --------------------------------------------------------
    # Hydrographic background
    # --------------------------------------------------------

    mesh = ax.pcolormesh(
        background["lon"].values,
        background["lat"].values,
        np.ma.masked_invalid(
            background.values
        ),
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=args.background_alpha,
        rasterized=True,
        zorder=1,
    )

    # --------------------------------------------------------
    # Gridlines
    # --------------------------------------------------------

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="0.45",
        alpha=0.35,
        linestyle="--",
        zorder=2,
    )

    # --------------------------------------------------------
    # SEANOE Eulerian fronts
    # --------------------------------------------------------

    front_handles = []

    for name in args.fronts:

        lon, lat = fronts[name]

        linestyle = FRONT_LINESTYLES[name]

        for lon_segment, lat_segment in split_at_dateline(
            lon,
            lat,
        ):

            line, = ax.plot(
                lon_segment,
                lat_segment,
                transform=ccrs.PlateCarree(),
                color=FRONT_COLOR,
                linestyle=linestyle,
                linewidth=args.front_width,
                alpha=args.front_alpha,
                zorder=10,
            )

            # White outline helps black fronts stay visible
            # over both dark and light backgrounds.
            line.set_path_effects(
                [
                    pe.Stroke(
                        linewidth=(
                            args.front_width
                            + args.front_halo
                        ),
                        foreground="white",
                    ),
                    pe.Normal(),
                ]
            )

        front_handles.append(
            Line2D(
                [],
                [],
                color=FRONT_COLOR,
                linestyle=linestyle,
                linewidth=args.front_width,
                label=name,
            )
        )

    # --------------------------------------------------------
    # ARGO flux cores
    # --------------------------------------------------------
    ax.scatter(
        argo["lon"],
        argo["lat"],
        transform=ccrs.PlateCarree(),
        marker=ARGO_MARKER,
        s=args.core_size,
        facecolor=ARGO_COLOR,
        edgecolor="black",
        linewidth=0.35,
        alpha=args.core_alpha*0.7,
        zorder=22,
    )

    # --------------------------------------------------------
    # Drifter flux cores
    # --------------------------------------------------------

    ax.scatter(
        drifter["lon"],
        drifter["lat"],
        transform=ccrs.PlateCarree(),
        marker=DRIFTER_MARKER,
        s=args.core_size,
        facecolor=DRIFTER_COLOR,
        edgecolor="black",
        linewidth=0.35,
        alpha=args.core_alpha,
        zorder=20,
    )

    # --------------------------------------------------------
    # Land/coastline
    # --------------------------------------------------------

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.90",
        edgecolor="black",
        linewidth=0.45,
        zorder=30,
    )

    ax.coastlines(
        linewidth=0.55,
        zorder=31,
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="vertical",
        shrink=0.76,
        pad=0.045,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=12,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    core_handles = [
        Line2D(
            [],
            [],
            marker=ARGO_MARKER,
            linestyle="none",
            markerfacecolor=ARGO_COLOR,
            markeredgecolor="black",
            markersize=7,
            label="ARGO flux core",
        ),

        Line2D(
            [],
            [],
            marker=DRIFTER_MARKER,
            linestyle="none",
            markerfacecolor=DRIFTER_COLOR,
            markeredgecolor="black",
            markersize=7,
            label="Drifter flux core",
        ),
    ]

    handles = (
        core_handles
        + front_handles
    )

    ax.legend(
        handles=handles,
        loc=args.legend_location,
        fontsize=8.5,
        frameon=True,
        framealpha=0.90,
        ncol=2,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if not args.no_title:

        if args.title is not None:
            title = args.title
        else:
            title = (
                "Lagrangian flux cores and "
                "Eulerian ACC fronts\n"
                f"{args.field.replace('_', ' ')}"
            )

        ax.set_title(
            title,
            fontsize=13,
            pad=12,
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {args.output}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Plot one ARGO hydrographic background field "
            "with ARGO/drifter flux cores and "
            "SEANOE ACC fronts."
        )
    )

    # --------------------------------------------------------
    # Required inputs
    # --------------------------------------------------------

    parser.add_argument(
        "--background",
        type=Path,
        required=True,
        help="sampled_map.nc",
    )

    parser.add_argument(
        "--field",
        required=True,
        help=(
            "Background variable, e.g. "
            "temp_mean, temp_gradient_magnitude, "
            "psal_mean, psal_gradient_magnitude."
        ),
    )

    parser.add_argument(
        "--argo-cores",
        type=Path,
        required=True,
        help="ARGO branch_cores.parquet",
    )

    parser.add_argument(
        "--drifter-cores",
        type=Path,
        required=True,
        help="Drifter branch_cores.parquet",
    )

    parser.add_argument(
        "--seanoe-fronts",
        type=Path,
        required=True,
        help=(
            "SEANOE_Altimetry_derived_"
            "Antarctic_Circumpolar_Current_fronts.nc"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path.",
    )

    # --------------------------------------------------------
    # Front selection
    # --------------------------------------------------------

    parser.add_argument(
        "--fronts",
        nargs="+",
        default=[
            "NB",
            "SAF",
            "PF",
            "SACCF",
            "SB",
        ],
        choices=[
            "NB",
            "SAF",
            "PF",
            "SACCF",
            "SB",
        ],
        help=(
            "SEANOE fronts to plot. "
            "Default: all five."
        ),
    )

    # --------------------------------------------------------
    # Geographic settings
    # --------------------------------------------------------

    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_LAT_MIN,
    )

    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_LAT_MAX,
    )

    parser.add_argument(
        "--central-longitude",
        type=float,
        default=0.0,
    )

    # --------------------------------------------------------
    # Background style
    # --------------------------------------------------------

    parser.add_argument(
        "--cmap",
        default=None,
        help=(
            "Override the default colormap "
            "for the selected field."
        ),
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Explicit colorbar minimum.",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Explicit colorbar maximum.",
    )

    parser.add_argument(
        "--percentile-min",
        type=float,
        default=None,
        help=(
            "Lower percentile for automatic "
            "color limits."
        ),
    )

    parser.add_argument(
        "--percentile-max",
        type=float,
        default=None,
        help=(
            "Upper percentile for automatic "
            "color limits. Default usually 99."
        ),
    )

    parser.add_argument(
        "--background-alpha",
        type=float,
        default=1.0,
    )

    # --------------------------------------------------------
    # Overlay style
    # --------------------------------------------------------

    parser.add_argument(
        "--core-size",
        type=float,
        default=DEFAULT_CORE_SIZE,
        help="ARGO/drifter core marker size.",
    )

    parser.add_argument(
        "--core-alpha",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--front-width",
        type=float,
        default=FRONT_WIDTH,
    )

    parser.add_argument(
        "--front-alpha",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--front-halo",
        type=float,
        default=1.2,
        help=(
            "Extra white outline width around "
            "Eulerian front lines."
        ),
    )

    # --------------------------------------------------------
    # Figure style
    # --------------------------------------------------------

    parser.add_argument(
        "--figsize",
        type=float,
        default=11.0,
        help="Square figure size in inches.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
    )

    parser.add_argument(
        "--legend-location",
        default="lower left",
    )

    parser.add_argument(
        "--title",
        default=None,
    )

    parser.add_argument(
        "--no-title",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    plot_map(args)


# ============================================================
# POWERSHELL EXAMPLES
# ============================================================

# ------------------------------------------------------------
# 1. TEMPERATURE MEAN
# ------------------------------------------------------------
#
# python -m lagrangian_fronts.plot_flux_cores_hydrography `
#     --background "data/sampled_map.nc" `
#     --field temp_mean `
#     --argo-cores "outputs/argo/analysis/flux_cores.parquet" `
#     --drifter-cores "outputs/drifters/analysis/flux_cores.parquet" `
#     --seanoe-fronts "data/acc_fronts.nc" `
#     --output "figures/flux_cores_temp_mean.png"
#
#
# ------------------------------------------------------------
# 2. TEMPERATURE GRADIENT MAGNITUDE
# ------------------------------------------------------------
#
# python -m lagrangian_fronts.plot_flux_cores_hydrography `
#     --background "data/sampled_map.nc" `
#     --field temp_gradient_magnitude `
#     --argo-cores "outputs/argo/analysis/flux_cores.parquet" `
#     --drifter-cores "outputs/drifters/analysis/flux_cores.parquet" `
#     --seanoe-fronts "data/acc_fronts.nc" `
#     --output "figures/flux_cores_temp_gradient.png"
#
#
# ------------------------------------------------------------
# 3. SALINITY MEAN
# ------------------------------------------------------------
#
# python -m lagrangian_fronts.plot_flux_cores_hydrography `
#     --background "data/sampled_map.nc" `
#     --field psal_mean `
#     --argo-cores "outputs/argo/analysis/flux_cores.parquet" `
#     --drifter-cores "outputs/drifters/analysis/flux_cores.parquet" `
#     --seanoe-fronts "data/acc_fronts.nc" `
#     --output "figures/flux_cores_psal_mean.png"
#
#
# ------------------------------------------------------------
# 4. SALINITY GRADIENT MAGNITUDE
# ------------------------------------------------------------
#
# python -m lagrangian_fronts.plot_flux_cores_hydrography `
#     --background "data/sampled_map.nc" `
#     --field psal_gradient_magnitude `
#     --argo-cores "outputs/argo/analysis/flux_cores.parquet" `
#     --drifter-cores "outputs/drifters/analysis/flux_cores.parquet" `
#     --seanoe-fronts "data/acc_fronts.nc" `
#     --output "figures/flux_cores_psal_gradient.png"
#
#
# ------------------------------------------------------------
# EXAMPLE: ONLY SAF, PF AND SACCF
# ------------------------------------------------------------
#
# python -m lagrangian_fronts.plot_flux_cores_hydrography `
#     --background "data/sampled_map.nc" `
#     --field temp_gradient_magnitude `
#     --argo-cores "outputs/argo/analysis/flux_cores.parquet" `
#     --drifter-cores "outputs/drifters/analysis/flux_cores.parquet" `
#     --seanoe-fronts "data/acc_fronts.nc" `
#     --fronts SAF PF SACCF `
#     --output "figures/flux_cores_temp_gradient_ACC_fronts.png"
#
#
# ------------------------------------------------------------
# EXAMPLE: MANUAL COLOR SCALE
# ------------------------------------------------------------
#
# python -m lagrangian_fronts.plot_flux_cores_hydrography `
#     --background "data/sampled_map.nc" `
#     --field psal_mean `
#     --argo-cores "outputs/argo/analysis/flux_cores.parquet" `
#     --drifter-cores "outputs/drifters/analysis/flux_cores.parquet" `
#     --seanoe-fronts "data/acc_fronts.nc" `
#     --vmin 34 `
#     --vmax 35 `
#     --output "figures/flux_cores_psal_manual_scale.png"