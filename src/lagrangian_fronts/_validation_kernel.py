"""Stage-7 independent global-gradient validation of Stage-6 flank points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .geometry import (
    GeographicGeometry,
    SpatialGeometry,
    _bilinear_supported_sample,
    _grid_array,
    _physical_cell_scales,
)


@dataclass(frozen=True)
class Stage7Fields:
    global_gradient_fields: pd.DataFrame
    global_gradient_dataset: xr.Dataset
    flank_gradient_comparison: pd.DataFrame
    unique_ridge_cell_side_comparison: pd.DataFrame
    summary: dict[str, Any]


GRADIENT_SAMPLE_FIELDS = (
    "G_perp_signed",
    "abs_G_perp",
    "G_parallel_signed",
    "abs_G_parallel",
    "gradient_magnitude",
    "F_perp_gradient",
)


def _neighbor(array: np.ndarray, axis: int, offset: int, periodic: bool) -> np.ndarray:
    if axis == 1 and periodic:
        return np.roll(array, -offset, axis=axis)
    output = np.full_like(array, np.nan, dtype=float)
    if axis == 0:
        if offset > 0:
            output[:-offset] = array[offset:]
        else:
            output[-offset:] = array[:offset]
    else:
        if offset > 0:
            output[:, :-offset] = array[:, offset:]
        else:
            output[:, -offset:] = array[:, :offset]
    return output


def _method_labels(
    centered: np.ndarray, one_sided: np.ndarray, prefix: str
) -> np.ndarray:
    values = np.full(centered.shape, f"{prefix}_undefined", dtype=object)
    values[one_sided] = f"{prefix}_one_sided"
    values[centered] = f"{prefix}_centered"
    return values


def compute_global_gradient_fields(
    cells: pd.DataFrame,
    grid: Any,
    *,
    geometry: SpatialGeometry | None = None,
    time_unit: str = "day",
    zero_tolerance: float = 1.0e-12,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Calculate raw, missing-aware physical gradients before Stage-6 use."""
    required = {
        "cell_id",
        "x_bin",
        "y_bin",
        "x",
        "y",
        "N_out_move",
        "U_out_all_magnitude_rate",
        "theta_mu_out",
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Stage-7 global field input lacks columns: {missing}")
    geometry = geometry or GeographicGeometry("WGS84", "km")
    scalar = _grid_array(cells, grid, "U_out_all_magnitude_rate")
    finite_center = np.isfinite(scalar)
    east = _neighbor(scalar, 1, 1, grid.periodic_x)
    west = _neighbor(scalar, 1, -1, grid.periodic_x)
    north = _neighbor(scalar, 0, 1, False)
    south = _neighbor(scalar, 0, -1, False)
    east_ok, west_ok = np.isfinite(east), np.isfinite(west)
    north_ok, south_ok = np.isfinite(north), np.isfinite(south)

    x2d = np.broadcast_to(
        grid.x_min + (np.arange(grid.nx) + 0.5) * grid.dx,
        scalar.shape,
    )
    y2d = np.broadcast_to(
        (grid.y_min + (np.arange(grid.ny) + 0.5) * grid.dy)[:, None],
        scalar.shape,
    )
    _, _, dx_centered_length = geometry.inverse(
        x2d - grid.dx, y2d, x2d + grid.dx, y2d
    )
    _, _, dx_east_length = geometry.inverse(x2d, y2d, x2d + grid.dx, y2d)
    _, _, dx_west_length = geometry.inverse(x2d - grid.dx, y2d, x2d, y2d)
    _, _, dy_centered_length = geometry.inverse(
        x2d, y2d - grid.dy, x2d, y2d + grid.dy
    )
    _, _, dy_north_length = geometry.inverse(x2d, y2d, x2d, y2d + grid.dy)
    _, _, dy_south_length = geometry.inverse(x2d, y2d - grid.dy, x2d, y2d)
    dx_centered_length = np.asarray(dx_centered_length)
    dx_east_length = np.asarray(dx_east_length)
    dx_west_length = np.asarray(dx_west_length)
    dy_centered_length = np.asarray(dy_centered_length)
    dy_north_length = np.asarray(dy_north_length)
    dy_south_length = np.asarray(dy_south_length)

    dS_dx = np.full_like(scalar, np.nan)
    dx_centered = finite_center & east_ok & west_ok
    dx_east = finite_center & east_ok & ~west_ok
    dx_west = finite_center & west_ok & ~east_ok
    dS_dx[dx_centered] = (east[dx_centered] - west[dx_centered]) / dx_centered_length[
        dx_centered
    ]
    dS_dx[dx_east] = (east[dx_east] - scalar[dx_east]) / dx_east_length[dx_east]
    dS_dx[dx_west] = (scalar[dx_west] - west[dx_west]) / dx_west_length[dx_west]

    dS_dy = np.full_like(scalar, np.nan)
    dy_centered = finite_center & north_ok & south_ok
    dy_north = finite_center & north_ok & ~south_ok
    dy_south = finite_center & south_ok & ~north_ok
    dS_dy[dy_centered] = (north[dy_centered] - south[dy_centered]) / dy_centered_length[
        dy_centered
    ]
    dS_dy[dy_north] = (north[dy_north] - scalar[dy_north]) / dy_north_length[dy_north]
    dS_dy[dy_south] = (scalar[dy_south] - south[dy_south]) / dy_south_length[dy_south]

    theta = _grid_array(cells, grid, "theta_mu_out")
    angle = np.deg2rad(theta)
    tangent_east, tangent_north = np.sin(angle), np.cos(angle)
    normal_east, normal_north = -tangent_north, tangent_east
    complete = np.isfinite(dS_dx) & np.isfinite(dS_dy) & np.isfinite(theta)
    g_perp = np.full_like(scalar, np.nan)
    g_parallel = np.full_like(scalar, np.nan)
    g_perp[complete] = (
        normal_east[complete] * dS_dx[complete]
        + normal_north[complete] * dS_dy[complete]
    )
    g_parallel[complete] = (
        tangent_east[complete] * dS_dx[complete]
        + tangent_north[complete] * dS_dy[complete]
    )
    magnitude = np.hypot(g_perp, g_parallel)
    fraction = np.full_like(scalar, np.nan)
    nonzero = complete & (magnitude > zero_tolerance)
    fraction[nonzero] = np.abs(g_perp[nonzero]) / magnitude[nonzero]
    dx_method = _method_labels(dx_centered, dx_east | dx_west, "dx")
    dy_method = _method_labels(dy_centered, dy_north | dy_south, "dy")

    output = cells.copy()
    output["S_flux"] = output.U_out_all_magnitude_rate
    flat_fields = {
        "dS_dx": dS_dx,
        "dS_dy": dS_dy,
        "G_perp_signed": g_perp,
        "abs_G_perp": np.abs(g_perp),
        "G_parallel_signed": g_parallel,
        "abs_G_parallel": np.abs(g_parallel),
        "gradient_magnitude": magnitude,
        "F_perp_gradient": fraction,
        "dx_method": dx_method,
        "dy_method": dy_method,
        "east_neighbor_available": east_ok,
        "west_neighbor_available": west_ok,
        "north_neighbor_available": north_ok,
        "south_neighbor_available": south_ok,
        "n_available_cardinal_neighbors": (
            east_ok.astype(int)
            + west_ok.astype(int)
            + north_ok.astype(int)
            + south_ok.astype(int)
        ),
    }
    indexes = (output.y_bin.to_numpy(np.int64), output.x_bin.to_numpy(np.int64))
    for name, values in flat_fields.items():
        output[name] = values[indexes]
    flags: list[str] = []
    for row in output.itertuples(index=False):
        row_flags: list[str] = []
        if not np.isfinite(row.S_flux):
            row_flags.append("flux_undefined")
        if row.dx_method == "dx_one_sided":
            row_flags.append("dx_one_sided")
        elif row.dx_method == "dx_undefined":
            row_flags.append("dx_undefined")
        if row.dy_method == "dy_one_sided":
            row_flags.append("dy_one_sided")
        elif row.dy_method == "dy_undefined":
            row_flags.append("dy_undefined")
        if not np.isfinite(row.theta_mu_out):
            row_flags.append("theta_mu_out_undefined")
        if (
            np.isfinite(row.gradient_magnitude)
            and row.gradient_magnitude <= zero_tolerance
        ):
            row_flags.append("gradient_zero_orientation_undefined")
        flags.append(";".join(row_flags))
    output["gradient_quality_flags"] = flags

    coords = {
        "y": grid.y_min + (np.arange(grid.ny) + 0.5) * grid.dy,
        "x": grid.x_min + (np.arange(grid.nx) + 0.5) * grid.dx,
    }
    dataset = xr.Dataset(coords=coords)
    numeric_names = (
        "S_flux",
        "dS_dx",
        "dS_dy",
        "G_perp_signed",
        "abs_G_perp",
        "G_parallel_signed",
        "abs_G_parallel",
        "gradient_magnitude",
        "F_perp_gradient",
        "theta_mu_out",
        "N_out_move",
        "n_available_cardinal_neighbors",
    )
    for name in numeric_names:
        dataset[name] = (("y", "x"), _grid_array(output, grid, name))
    for name in ("dx_method", "dy_method", "gradient_quality_flags"):
        values = np.full((grid.ny, grid.nx), "", dtype=object)
        values[indexes] = output[name].astype(str).to_numpy()
        dataset[name] = (("y", "x"), values.astype(str))
    for name in (
        "dS_dx",
        "dS_dy",
        "G_perp_signed",
        "abs_G_perp",
        "G_parallel_signed",
        "abs_G_parallel",
        "gradient_magnitude",
    ):
        dataset[name].attrs["units"] = f"{time_unit}-1"
    dataset["S_flux"].attrs["units"] = (
        f"{geometry.length_unit} {time_unit}-1"
    )
    dataset.attrs.update(
        scalar_field="raw |U_out_all|",
        differentiation=(
            f"missing-aware {geometry.coordinate_system} physical finite differences"
        ),
        unsupported_values="NaN; never zero-filled",
        stage6_used_in_field_construction="false",
    )
    return output, dataset


def _sample_gradient(
    global_cells: pd.DataFrame,
    grid: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    interpolation_weight_tolerance: float,
    direct_sample_atol_grid_cells: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    support_grid = np.isfinite(_grid_array(global_cells, grid, "abs_G_perp"))
    sampled: dict[str, np.ndarray] = {}
    boundary_any = np.zeros(len(x), dtype=bool)
    missing_any = np.zeros(len(x), dtype=bool)
    for name in GRADIENT_SAMPLE_FIELDS:
        values, boundary, missing = _bilinear_supported_sample(
            _grid_array(global_cells, grid, name),
            support_grid,
            x,
            y,
            grid,
            weight_tolerance=interpolation_weight_tolerance,
        )
        sampled[name] = values
        boundary_any |= boundary
        missing_any |= missing
    if grid.periodic_x:
        x = ((x - grid.x_min) % (grid.x_max - grid.x_min)) / grid.dx - 0.5
    else:
        x = (x - grid.x_min) / grid.dx - 0.5
    y = (y - grid.y_min) / grid.dy - 0.5
    direct = np.isclose(
        x, np.round(x), atol=direct_sample_atol_grid_cells
    ) & np.isclose(y, np.round(y), atol=direct_sample_atol_grid_cells)
    available = np.isfinite(sampled["abs_G_perp"])
    sample_class = np.full(len(x), "gradient_sample_unavailable", dtype=object)
    sample_class[available & ~direct] = "gradient_sample_interpolated"
    sample_class[available & direct] = "gradient_sample_direct"
    return sampled, sample_class, boundary_any, missing_any


def _local_gradient_context(
    points: pd.DataFrame,
    global_cells: pd.DataFrame,
    geometry: SpatialGeometry,
    *,
    search_radius_grid_scales: float,
    background_radius_grid_scales: float,
) -> pd.DataFrame:
    supported = global_cells.loc[
        global_cells.abs_G_perp.notna(), ["cell_id", "x", "y", "abs_G_perp"]
    ]
    records: list[dict[str, Any]] = []
    for point in points.itertuples(index=False):
        _, _, flank_distance = geometry.inverse(
            np.full(len(supported), point.flank_x),
            np.full(len(supported), point.flank_y),
            supported.x.to_numpy(float),
            supported.y.to_numpy(float),
        )
        flank_distance_length = np.asarray(flank_distance)
        local = supported.loc[
            flank_distance_length
            <= search_radius_grid_scales * point.grid_effective_scale_length
        ].copy()
        local_distances = flank_distance_length[
            flank_distance_length
            <= search_radius_grid_scales * point.grid_effective_scale_length
        ]
        if local.empty:
            local_max = max_distance = max_x = max_y = np.nan
        else:
            maximum = float(local.abs_G_perp.max())
            choices = np.flatnonzero(
                np.isclose(local.abs_G_perp.to_numpy(float), maximum)
            )
            selected = choices[np.argmin(local_distances[choices])]
            row = local.iloc[selected]
            local_max = maximum
            max_distance = float(local_distances[selected])
            max_x, max_y = float(row.x), float(row.y)

        _, _, core_distance = geometry.inverse(
            np.full(len(supported), point.refined_core_x),
            np.full(len(supported), point.refined_core_y),
            supported.x.to_numpy(float),
            supported.y.to_numpy(float),
        )
        background = supported.loc[
            np.asarray(core_distance)
            <= background_radius_grid_scales * point.grid_effective_scale_length,
            "abs_G_perp",
        ]
        percentile = (
            float((background <= point.abs_G_perp_at_flank).mean())
            if len(background) and np.isfinite(point.abs_G_perp_at_flank)
            else np.nan
        )
        records.append(
            {
                "comparison_record_id": point.comparison_record_id,
                "local_max_abs_G_perp": local_max,
                "local_gradient_max_x": max_x,
                "local_gradient_max_y": max_y,
                "distance_to_local_gradient_max_length": max_distance,
                "distance_to_local_gradient_max_L_eff": (
                    max_distance / point.grid_effective_scale_length
                    if np.isfinite(max_distance) and point.grid_effective_scale_length > 0
                    else np.nan
                ),
                "local_abs_G_perp_percentile": percentile,
                "n_local_background_cells": len(background),
            }
        )
    return pd.DataFrame.from_records(records)


def _segment_comparison(
    global_cells: pd.DataFrame,
    flanks: pd.DataFrame,
    sections: pd.DataFrame,
    grid: Any,
    config: Any,
    geometry: SpatialGeometry,
) -> pd.DataFrame:
    context_columns = [
        "section_id",
        "ridge_type",
        "stage5_missing_side",
        "ridge_x",
        "ridge_y",
        "refined_core_x",
        "refined_core_y",
        "grid_effective_scale_length",
        "high_local_curvature_turning",
        "R1_out_center",
        "n_observable_flanks",
    ]
    output = flanks.merge(sections[context_columns], on="section_id", how="left")
    output = output.rename(
        columns={
            "cell_id": "ridge_cell_id",
            "candidate_x": "flank_x",
            "candidate_y": "flank_y",
            "candidate_distance_length": "flank_distance_length",
            "absolute_drop": "absolute_flux_loss",
            "relative_drop": "relative_flux_loss",
            "along_branch_persistence": "stage6_persistence",
        }
    )
    output.insert(
        0,
        "comparison_record_id",
        [f"segment_flank_{index:05d}" for index in range(len(output))],
    )
    flank_sample, sample_class, boundary, missing = _sample_gradient(
        global_cells,
        grid,
        output.flank_x.to_numpy(float),
        output.flank_y.to_numpy(float),
        interpolation_weight_tolerance=config.interpolation_weight_tolerance,
        direct_sample_atol_grid_cells=config.direct_sample_atol_grid_cells,
    )
    core_sample, core_class, _, _ = _sample_gradient(
        global_cells,
        grid,
        output.refined_core_x.to_numpy(float),
        output.refined_core_y.to_numpy(float),
        interpolation_weight_tolerance=config.interpolation_weight_tolerance,
        direct_sample_atol_grid_cells=config.direct_sample_atol_grid_cells,
    )
    rename = {
        "G_perp_signed": "G_perp_at_flank",
        "abs_G_perp": "abs_G_perp_at_flank",
        "G_parallel_signed": "G_parallel_at_flank",
        "abs_G_parallel": "abs_G_parallel_at_flank",
        "gradient_magnitude": "gradient_magnitude_at_flank",
        "F_perp_gradient": "F_perp_gradient_at_flank",
    }
    for source, target in rename.items():
        output[target] = flank_sample[source]
    output["abs_G_perp_at_core"] = core_sample["abs_G_perp"]
    output["gradient_sample_class"] = sample_class
    output["core_gradient_sample_class"] = core_class
    output["gradient_observability"] = np.where(
        np.isfinite(output.abs_G_perp_at_flank),
        "gradient_observable",
        "gradient_not_observable",
    )
    output["gradient_sample_boundary"] = boundary
    output["gradient_sample_missing_support"] = missing
    ratio = output.abs_G_perp_at_flank / (
        output.abs_G_perp_at_core + config.core_gradient_ratio_epsilon
    )
    unstable = output.abs_G_perp_at_core.le(config.core_gradient_ratio_epsilon)
    output["flank_to_core_abs_G_perp_ratio"] = ratio.mask(unstable)
    output["core_gradient_ratio_unstable"] = unstable
    local = _local_gradient_context(
        output,
        global_cells,
        geometry,
        search_radius_grid_scales=config.gradient_search_radius_grid_scales,
        background_radius_grid_scales=config.local_background_radius_grid_scales,
    )
    output = output.merge(local, on="comparison_record_id", how="left")
    output["quality_flags"] = output.apply(
        lambda row: ";".join(
            flag
            for flag, condition in (
                (str(row.get("quality_flags", "")), bool(row.get("quality_flags", ""))),
                (
                    "gradient_not_observable",
                    row.gradient_observability == "gradient_not_observable",
                ),
                ("gradient_sample_boundary", bool(row.gradient_sample_boundary)),
                (
                    "gradient_sample_missing_support",
                    bool(row.gradient_sample_missing_support),
                ),
                (
                    "core_gradient_not_observable",
                    not np.isfinite(row.abs_G_perp_at_core),
                ),
                (
                    "core_gradient_ratio_unstable",
                    bool(row.core_gradient_ratio_unstable),
                ),
            )
            if condition and flag
        ),
        axis=1,
    )
    return output


def _unique_comparison(segment: pd.DataFrame, config: Any) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    median_fields = [
        "flank_x",
        "flank_y",
        "flank_distance_length",
        "absolute_flux_loss",
        "relative_flux_loss",
        "G_perp_at_flank",
        "abs_G_perp_at_flank",
        "G_parallel_at_flank",
        "abs_G_parallel_at_flank",
        "gradient_magnitude_at_flank",
        "F_perp_gradient_at_flank",
        "abs_G_perp_at_core",
        "flank_to_core_abs_G_perp_ratio",
        "local_max_abs_G_perp",
        "distance_to_local_gradient_max_length",
        "distance_to_local_gradient_max_L_eff",
        "local_abs_G_perp_percentile",
        "grid_effective_scale_length",
        "R1_out_center",
    ]
    for (cell_id, side), group in segment.groupby(["ridge_cell_id", "side"], sort=True):
        first = group.iloc[0]
        distance_spread = float(
            group.flank_distance_length.max() - group.flank_distance_length.min()
        )
        scale = float(group.grid_effective_scale_length.median())
        record: dict[str, Any] = {
            "unique_comparison_id": f"cell{int(cell_id):05d}_{side}",
            "ridge_cell_id": int(cell_id),
            "side": side,
            "ridge_type": first.ridge_type,
            "stage5_missing_side": first.stage5_missing_side,
            "n_segment_records": len(group),
            "comparison_record_ids": ";".join(group.comparison_record_id),
            "component_ids": ";".join(sorted(set(group.component_id.astype(str)))),
            "segment_ids": ";".join(sorted(set(group.segment_id.astype(str)))),
            "section_ids": ";".join(sorted(set(group.section_id.astype(str)))),
            "candidate_distance_spread_length": distance_spread,
            "candidate_distance_spread_L_eff": distance_spread / scale
            if scale > 0
            else np.nan,
            "duplicate_flank_disagreement": bool(
                len(group) > 1
                and distance_spread > config.duplicate_disagreement_grid_scales * scale
            ),
            "stage6_persistence": bool(group.stage6_persistence.mean() >= 0.5),
            "stage6_persistence_fraction": float(group.stage6_persistence.mean()),
            "nearby_branch_contamination": bool(
                group.nearby_branch_contamination.any()
            ),
            "high_local_curvature_turning": bool(
                group.high_local_curvature_turning.any()
            ),
            "gradient_observability": (
                "gradient_observable"
                if group.abs_G_perp_at_flank.notna().any()
                else "gradient_not_observable"
            ),
            "gradient_sample_classes": ";".join(
                sorted(set(group.gradient_sample_class))
            ),
        }
        for field in median_fields:
            record[field] = float(group[field].median())
        flags = sorted(
            {
                flag
                for value in group.quality_flags.astype(str)
                for flag in value.split(";")
                if flag
            }
        )
        if record["duplicate_flank_disagreement"]:
            flags.append("duplicate_flank_disagreement")
        record["quality_flags"] = ";".join(sorted(set(flags)))
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_stage7_fields(
    stage4_cells: pd.DataFrame,
    stage6_flanks: pd.DataFrame,
    stage6_sections: pd.DataFrame,
    grid: Any,
    *,
    config: Any,
    geometry: SpatialGeometry | None = None,
    time_unit: str = "day",
    precomputed_global: tuple[pd.DataFrame, xr.Dataset] | None = None,
    primary_experiment_id: str,
) -> Stage7Fields:
    """Construct the independent field, then compare boundary-aware q90 flanks."""
    geometry = geometry or GeographicGeometry("WGS84", "km")
    if precomputed_global is None:
        global_cells, dataset = compute_global_gradient_fields(
            stage4_cells,
            grid,
            geometry=geometry,
            time_unit=time_unit,
            zero_tolerance=config.gradient_zero_tolerance,
        )
    else:
        global_cells, dataset = precomputed_global
    primary_id = primary_experiment_id
    flanks = stage6_flanks.loc[stage6_flanks.experiment_id.eq(primary_id)].copy()
    sections = stage6_sections.loc[stage6_sections.experiment_id.eq(primary_id)].copy()
    segment = _segment_comparison(
        global_cells, flanks, sections, grid, config, geometry
    )
    unique = _unique_comparison(segment, config)
    evaluable = unique.loc[unique.abs_G_perp_at_flank.notna()]
    paired = evaluable.loc[evaluable.abs_G_perp_at_core.notna()]
    two_flank = unique.groupby("ridge_cell_id").filter(
        lambda group: {"left", "right"} <= set(group.side)
    )
    sign_groups = (
        two_flank.groupby("ridge_cell_id")
        .filter(lambda group: group.G_perp_at_flank.notna().all())
        .groupby("ridge_cell_id")
    )
    opposite = []
    for _, group in sign_groups:
        left = group.loc[group.side.eq("left"), "G_perp_at_flank"].median()
        right = group.loc[group.side.eq("right"), "G_perp_at_flank"].median()
        opposite.append(bool(left * right < 0))
    global_supported = global_cells.loc[global_cells.abs_G_perp.notna()]
    global_levels = {
        f"q{int(q * 100)}": float(global_supported.abs_G_perp.quantile(q))
        for q in (0.5, 0.75, 0.9, 0.95)
    }
    _, _, global_effective_scale = _physical_cell_scales(
        global_supported, grid, geometry
    )
    global_supported = global_supported.copy()
    global_supported["grid_effective_scale_length"] = global_effective_scale
    strong_representation: dict[str, Any] = {}
    for label in ("q90", "q95"):
        strong = global_supported.loc[
            global_supported.abs_G_perp.ge(global_levels[label])
        ]
        represented = 0
        for row in strong.itertuples(index=False):
            if unique.empty:
                continue
            _, _, distances = geometry.inverse(
                np.full(len(unique), row.x),
                np.full(len(unique), row.y),
                unique.flank_x.to_numpy(float),
                unique.flank_y.to_numpy(float),
            )
            represented += int(
                np.nanmin(np.asarray(distances))
                <= row.grid_effective_scale_length
            )
        strong_representation[label] = {
            "n_global_cells": len(strong),
            "n_with_stage6_flank_within_1_L_eff": int(represented),
            "fraction_with_stage6_flank_within_1_L_eff": (
                represented / len(strong) if len(strong) else np.nan
            ),
        }
    summary: dict[str, Any] = {
        "baseline_stage6_experiment": primary_id,
        "scalar_field": "raw U_out_all_magnitude_rate",
        "stage6_used_in_global_gradient_construction": False,
        "zero_filled_cells": 0,
        "global_cells": len(global_cells),
        "global_flux_defined_cells": int(global_cells.S_flux.notna().sum()),
        "global_complete_gradient_cells": int(global_cells.abs_G_perp.notna().sum()),
        "dx_centered_cells": int(global_cells.dx_method.eq("dx_centered").sum()),
        "dx_one_sided_cells": int(global_cells.dx_method.eq("dx_one_sided").sum()),
        "dx_undefined_cells": int(global_cells.dx_method.eq("dx_undefined").sum()),
        "dy_centered_cells": int(global_cells.dy_method.eq("dy_centered").sum()),
        "dy_one_sided_cells": int(global_cells.dy_method.eq("dy_one_sided").sum()),
        "dy_undefined_cells": int(global_cells.dy_method.eq("dy_undefined").sum()),
        "global_abs_G_perp_descriptive_levels": global_levels,
        "segment_side_flanks": len(segment),
        "unique_ridge_cell_side_flanks": len(unique),
        "duplicate_segment_side_records": int(len(segment) - len(unique)),
        "unique_duplicate_flank_disagreements": int(
            unique.duplicate_flank_disagreement.sum()
        ),
        "unique_gradient_evaluable_flanks": len(evaluable),
        "unique_gradient_not_observable_flanks": int(
            unique.gradient_observability.eq("gradient_not_observable").sum()
        ),
        "median_abs_G_perp_at_flank": float(evaluable.abs_G_perp_at_flank.median()),
        "median_abs_G_perp_at_core": float(paired.abs_G_perp_at_core.median()),
        "median_local_abs_G_perp_percentile": float(
            evaluable.local_abs_G_perp_percentile.median()
        ),
        "fraction_within_0_5_L_eff_of_local_max": float(
            evaluable.distance_to_local_gradient_max_L_eff.le(0.5).mean()
        ),
        "fraction_within_1_0_L_eff_of_local_max": float(
            evaluable.distance_to_local_gradient_max_L_eff.le(1.0).mean()
        ),
        "median_distance_to_local_max_length": float(
            evaluable.distance_to_local_gradient_max_length.median()
        ),
        "median_distance_to_local_max_L_eff": float(
            evaluable.distance_to_local_gradient_max_L_eff.median()
        ),
        "fraction_flank_gradient_exceeds_core": float(
            (paired.abs_G_perp_at_flank > paired.abs_G_perp_at_core).mean()
        ),
        "opposite_left_right_signed_gradient_fraction": float(np.mean(opposite))
        if opposite
        else np.nan,
        "n_ridge_cells_with_evaluable_left_right_signs": len(opposite),
        "spearman_flux_loss_vs_abs_G_perp": float(
            evaluable.absolute_flux_loss.corr(
                evaluable.abs_G_perp_at_flank, method="spearman"
            )
        ),
        "strong_global_gradient_representation": strong_representation,
        "continuous_front_lines_created": False,
        "final_gradient_threshold_selected": False,
        "permeability_implemented": False,
        "physical_identification_performed": False,
        "geographic_filters_applied": False,
    }
    return Stage7Fields(
        global_gradient_fields=global_cells,
        global_gradient_dataset=dataset,
        flank_gradient_comparison=segment,
        unique_ridge_cell_side_comparison=unique,
        summary=summary,
    )
