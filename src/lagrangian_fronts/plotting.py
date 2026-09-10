"""Publication-ready maps for flux and directional structure analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MatplotlibPath

from .config import AnalysisConfig
from .cores import CoreSolution
from .directional_cores import DirectionalCoreSolution
from .directional_fronts import DirectionalFrontSolution
from .fronts import FrontSolution
from .validation import ValidationSolution

CORE_MARKER_STYLES = {
    "two_sided": ("o", "tab:blue", "Flux core (two-sided observed)"),
    "one_sided": ("^", "tab:orange", "Flux core (one-sided observed)"),
}
FRONT_MARKER_STYLES = {
    "left": (">", "cyan", "Left flux front"),
    "right": ("<", "magenta", "Right flux front"),
}
DIRECTIONAL_CORE_MARKER_STYLES = {
    "two_sided": ("o", "tab:green", "Directional core (two-sided ridge)"),
    "one_sided": ("^", "tab:orange", "Directional core (one-sided ridge)"),
}
DIRECTIONAL_FRONT_MARKER_STYLES = {
    "left": (">", "cyan", "Left directional front"),
    "right": ("<", "magenta", "Right directional front"),
}


def _finite_percentile_max(values, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    maximum = float(np.percentile(finite, percentile))
    return maximum if maximum > 0 else 1.0


def _style_quiver_key_label(key) -> None:
    """Keep a quiver-key label readable over a mapped field."""
    key.text.set_bbox(
        {"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5}
    )


def _projection(config):
    import cartopy.crs as ccrs

    data = ccrs.PlateCarree()
    projection = (
        ccrs.SouthPolarStereo(central_longitude=config.central_longitude)
        if config.projection == "SouthPolarStereo"
        else data
    )
    return projection, data


def create_standard_figures(
    cells,
    cores: CoreSolution,
    fronts: FrontSolution,
    config: AnalysisConfig,
    output_dir: Path,
    *,
    directional_cores: DirectionalCoreSolution,
    directional_fronts: DirectionalFrontSolution,
    validation: ValidationSolution | None = None,
) -> list[Path]:
    """Create flux/directional maps and an optional validation map."""
    if not config.plotting.enabled:
        return []
    geographic = config.geometry.coordinate_system == "geographic"
    projection, data_crs = _projection(config.plotting) if geographic else (None, None)
    data_transform = {"transform": data_crs} if geographic else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    x_edges = np.linspace(
        config.grid.x_min, config.grid.x_max, config.grid.nx + 1
    )
    y_edges = np.linspace(
        config.grid.y_min, config.grid.y_max, config.grid.ny + 1
    )
    support = cells.N_out_move.ge(config.statistics.min_moving_support)
    created: list[Path] = []

    def grid_array(field):
        values = np.full((config.grid.ny, config.grid.nx), np.nan)
        rows = cells.loc[support, ["y_bin", "x_bin", field]]
        values[rows.y_bin.to_numpy(int), rows.x_bin.to_numpy(int)] = rows[field]
        return values

    def base_axis():
        if geographic:
            figure, axis = plt.subplots(
                figsize=(9, 8), subplot_kw={"projection": projection}
            )
            axis.set_extent(
                [
                    config.grid.x_min,
                    config.grid.x_max,
                    config.grid.y_min,
                    config.grid.y_max,
                ],
                crs=data_crs,
            )
            if config.plotting.draw_coastlines:
                axis.coastlines(linewidth=0.5)
            if (
                config.plotting.circular_boundary
                and config.plotting.projection == "SouthPolarStereo"
            ):
                angles = np.linspace(0, 2 * np.pi, 128)
                boundary = np.column_stack(
                    [0.5 + 0.5 * np.sin(angles), 0.5 + 0.5 * np.cos(angles)]
                )
                axis.set_boundary(MatplotlibPath(boundary), transform=axis.transAxes)
            axis.gridlines(linewidth=0.3, alpha=0.35)
        else:
            figure, axis = plt.subplots(figsize=(9, 8))
            axis.set_xlim(config.grid.x_min, config.grid.x_max)
            axis.set_ylim(config.grid.y_min, config.grid.y_max)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(f"x [{config.geometry.length_unit}]")
            axis.set_ylabel(f"y [{config.geometry.length_unit}]")
            axis.grid(linewidth=0.3, alpha=0.35)
        return figure, axis

    def add_flux_background(axis, *, cmap="viridis", max_percentile=99.0):
        values = grid_array("U_out_all_magnitude_rate")
        return axis.pcolormesh(
            x_edges,
            y_edges,
            values,
            **data_transform,
            shading="flat",
            cmap=cmap,
            vmin=0,
            vmax=_finite_percentile_max(values, max_percentile),
        )

    def add_flux_vectors(axis):
        stride = config.plotting.vector_stride_cells
        arrows = cells.loc[
            support
            & cells.x_bin.mod(stride).eq(0)
            & cells.y_bin.mod(stride).eq(0)
            & cells.U_out_all_x_rate.notna()
            & cells.U_out_all_y_rate.notna()
        ]
        quiver = axis.quiver(
            arrows.x,
            arrows.y,
            arrows.U_out_all_x_rate,
            arrows.U_out_all_y_rate,
            **data_transform,
            color="black",
            width=0.0022,
            headwidth=3.5,
            zorder=3,
        )
        key = axis.quiverkey(
            quiver,
            0.80,
            0.035,
            config.plotting.vector_reference,
            (
                f"{config.plotting.vector_reference:g} "
                f"{config.geometry.length_unit} {config.matrix.time_unit}$^{{-1}}$ "
                "flux vector"
            ),
            labelpos="N",
            coordinates="axes",
        )
        _style_quiver_key_label(key)

    def add_directional_background(axis, *, cmap="viridis"):
        values = grid_array("D_out_all_magnitude")
        return axis.pcolormesh(
            x_edges,
            y_edges,
            values,
            **data_transform,
            shading="flat",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

    def add_directional_vectors(axis, *, add_key=True):
        stride = config.plotting.vector_stride_cells
        arrows = cells.loc[
            support
            & cells.x_bin.mod(stride).eq(0)
            & cells.y_bin.mod(stride).eq(0)
            & cells.D_out_all_x.notna()
            & cells.D_out_all_y.notna()
        ]
        quiver = axis.quiver(
            arrows.x,
            arrows.y,
            arrows.D_out_all_x,
            arrows.D_out_all_y,
            **data_transform,
            color="black",
            width=0.0022,
            headwidth=3.5,
            zorder=3,
        )
        if add_key:
            reference = config.plotting.directional_vector_reference
            key = axis.quiverkey(
                quiver,
                0.72,
                0.08,
                reference,
                rf"$|D|={reference:g}$ directional vector (dimensionless; not velocity)",
                labelpos="N",
                coordinates="axes",
                fontproperties={"size": 8},
            )
            _style_quiver_key_label(key)

    def save(figure, filename):
        path = output_dir / filename
        figure.savefig(path, dpi=config.plotting.dpi, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    figure, axis = base_axis()
    mesh = add_flux_background(axis)
    add_flux_vectors(axis)
    rate_label = (
        f"|U_out,all| [{config.geometry.length_unit} "
        f"{config.matrix.time_unit}$^{{-1}}$]"
    )
    figure.colorbar(mesh, ax=axis, shrink=0.7, label=rate_label)
    axis.set_title("Lagrangian displacement flux")
    save(figure, "01_flux_vectors.png")

    scalar_maps = (
        (
            "R1_out",
            "02_R1.png",
            "Outgoing first angular harmonic $R_1$",
            "R1",
            "viridis",
        ),
        (
            "R2_out",
            "03_R2.png",
            "Outgoing second angular harmonic $R_2$",
            "R2",
            "viridis",
        ),
        (
            "angular_entropy_out",
            "04_angular_entropy.png",
            "Normalized outgoing angular entropy",
            "Normalized angular entropy",
            "magma",
        ),
    )
    for field, filename, title, label, cmap in scalar_maps:
        figure, axis = base_axis()
        mesh = axis.pcolormesh(
            x_edges,
            y_edges,
            grid_array(field),
            **data_transform,
            shading="flat",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        figure.colorbar(mesh, ax=axis, shrink=0.7, label=label)
        axis.set_title(title)
        save(figure, filename)

    figure, axis = base_axis()
    mesh = add_flux_background(
        axis,
        cmap="Greys",
        max_percentile=config.plotting.structure_map_max_percentile,
    )
    add_flux_vectors(axis)
    for ridge_type, (marker, color, label) in CORE_MARKER_STYLES.items():
        rows = cores.cores.loc[cores.cores.ridge_type.eq(ridge_type)]
        if not rows.empty:
            axis.scatter(
                rows.x,
                rows.y,
                **data_transform,
                s=15,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.25,
                label=label,
                zorder=5,
            )
    detected = fronts.fronts.loc[fronts.fronts.front_detected]
    for side, (marker, color, label) in FRONT_MARKER_STYLES.items():
        rows = detected.loc[detected.side.eq(side)]
        if not rows.empty:
            axis.scatter(
                rows.front_x,
                rows.front_y,
                **data_transform,
                s=18,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                label=label,
                zorder=6,
            )
    axis.legend(loc="lower left", fontsize=7, framealpha=0.9)
    figure.colorbar(mesh, ax=axis, shrink=0.7, label=rate_label)
    axis.set_title("Lagrangian flux cores and probable flux fronts")
    save(figure, "05_flux_cores_and_fronts.png")

    figure, axis = base_axis()
    mesh = add_directional_background(axis)
    add_directional_vectors(axis)
    figure.colorbar(
        mesh,
        ax=axis,
        shrink=0.7,
        label=r"$|D_{out,all}| = P_{move}R_1$ [dimensionless]",
    )
    axis.set_title("Lagrangian directional organization field (distance-free)")
    save(figure, "06_directional_vectors.png")

    figure, axis = base_axis()
    mesh = add_directional_background(axis, cmap="Greys")
    add_directional_vectors(axis, add_key=False)
    candidate_rows = directional_cores.candidate_diagnostics
    if not candidate_rows.empty:
        axis.scatter(
            candidate_rows.x,
            candidate_rows.y,
            **data_transform,
            s=9,
            marker="s",
            color="tab:gray",
            alpha=0.35,
            linewidths=0,
            label="Directional candidate area",
            zorder=4,
        )
    for ridge_type, (marker, color, label) in DIRECTIONAL_CORE_MARKER_STYLES.items():
        rows = directional_cores.cores.loc[
            directional_cores.cores.ridge_type.eq(ridge_type)
        ]
        if not rows.empty:
            axis.scatter(
                rows.x,
                rows.y,
                **data_transform,
                s=15,
                marker=marker,
                color=color,
                edgecolors="white" if marker != "x" else None,
                linewidths=0.25,
                label=label,
                zorder=5,
            )
    detected_directional = directional_fronts.fronts.loc[
        directional_fronts.fronts.front_detected
    ]
    for side, (marker, color, label) in DIRECTIONAL_FRONT_MARKER_STYLES.items():
        rows = detected_directional.loc[detected_directional.side.eq(side)]
        if not rows.empty:
            axis.scatter(
                rows.front_x,
                rows.front_y,
                **data_transform,
                s=18,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                label=label,
                zorder=6,
            )
    axis.legend(loc="lower left", fontsize=7, framealpha=0.9)
    figure.colorbar(
        mesh,
        ax=axis,
        shrink=0.7,
        label=r"$|D_{out,all}| = P_{move}R_1$ [dimensionless]",
    )
    axis.set_title("Directional cores and probable directional fronts")
    save(figure, "07_directional_cores_and_fronts.png")

    if validation is not None and config.plotting.debug_plots:
        values = np.full((config.grid.ny, config.grid.nx), np.nan)
        frame = validation.global_fields
        values[frame.y_bin.to_numpy(int), frame.x_bin.to_numpy(int)] = frame[
            "abs_G_perp"
        ]
        figure, axis = base_axis()
        mesh = axis.pcolormesh(
            x_edges,
            y_edges,
            values,
            **data_transform,
            shading="flat",
            cmap="magma",
            vmin=0,
            vmax=_finite_percentile_max(
                values, config.plotting.structure_map_max_percentile
            ),
        )
        rows = validation.validation
        axis.scatter(
            rows.flank_x,
            rows.flank_y,
            **data_transform,
            s=12,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.6,
            label="Probable flux front",
        )
        axis.legend(loc="lower left", fontsize=7, framealpha=0.9)
        figure.colorbar(
            mesh,
            ax=axis,
            shrink=0.7,
            label=f"|G_perp| [{config.matrix.time_unit}$^{{-1}}$]",
        )
        axis.set_title("Optional cross-stream-gradient validation")
        save(figure, "debug_gradient_validation.png")

    return created
