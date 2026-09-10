from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .config import GridConfig, ValidationConfig
from .geometry import (
    NEIGHBOR_OFFSETS_8,
    GeographicGeometry,
    SpatialGeometry,
    signed_angle_difference,
)


@dataclass(frozen=True)
class GeometryConfig(GeographicGeometry):
    """Legacy internal test convenience; production uses configured backends."""

    ellipsoid: str = "WGS84"
    length_unit: str = "km"
    coordinate_system: str = "geographic"

REQUIRED_COLUMNS = (
    "start_x_bin",
    "start_y_bin",
    "end_x_bin",
    "end_y_bin",
    "start_x_center",
    "start_y_center",
    "end_x_center",
    "end_y_center",
    "transition_count",
    "transition_probability",
)
KEY_COLUMNS = (
    "start_x_bin",
    "start_y_bin",
    "end_x_bin",
    "end_y_bin",
)
COUNT_FIELDS = ("N_out_total", "N_out_move", "N_in_total", "N_in_move")


@dataclass(frozen=True)
class Stage1Config:
    primary_visualization_min_moving_count: int = 10
    sensitivity_visualization_min_moving_count: int = 20
    direction_zero_tolerance: float = 1.0e-12
    distance_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 0.95)
    diagnostic_strong_flux_quantile: float = 0.95
    diagnostic_reduction_quantile: float = 0.99


@dataclass(frozen=True)
class Stage2Config:
    angular_bins: int = 36
    harmonic_zero_tolerance: float = 1.0e-12
    high_R1: float = 0.8
    low_R1: float = 0.5
    high_R2: float = 0.8
    large_delta_theta_degrees: float = 30.0
    split_R1_min: float = 0.4
    split_R1_max: float = 0.85
    split_R2_max: float = 0.3


@dataclass(frozen=True)
class Stage3Config:
    angular_bins: int = 36
    harmonic_zero_tolerance: float = 1.0e-12
    diagnostic_clean_alignment_min: float = 0.8
    diagnostic_poor_alignment_max: float = 0.0
    diagnostic_alignment_difference_min: float = 0.5
    diagnostic_large_delta_theta_degrees: float = 30.0


@dataclass(frozen=True)
class Stage3BConfig:
    representative_cells_per_category: int = 3
    diagnostic_high_neighborhood_consistency: float = 0.8
    diagnostic_poor_neighborhood_consistency: float = 0.0
    diagnostic_poor_alignment_max: float = 0.0
    diagnostic_large_representation_difference: float = 0.5


@dataclass(frozen=True)
class Stage4Config:
    flux_percentiles: tuple[float, ...] = (0.9, 0.95, 0.99)
    substantial_turn_degrees: float = 60.0
    low_alignment_turn_degrees: float = 90.0
    reversal_like_turn_degrees: float = 150.0
    direction_difference_review_degrees: float = 20.0
    representative_cells_per_category: int = 3
    vector_decimation: int = 5


@dataclass(frozen=True)
class ValidationResult:
    links: pd.DataFrame
    rows: pd.DataFrame
    invalid_links: pd.DataFrame
    summary: dict[str, Any]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class SupportResult:
    cells: pd.DataFrame
    dataset: xr.Dataset
    coverage: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class Stage1Fields:
    links: pd.DataFrame
    cells: pd.DataFrame
    dataset: xr.Dataset
    summary: dict[str, Any]


@dataclass(frozen=True)
class Stage2Fields:
    cells: pd.DataFrame
    dataset: xr.Dataset
    summary: dict[str, Any]


@dataclass(frozen=True)
class Stage3Fields:
    links: pd.DataFrame
    cells: pd.DataFrame
    dataset: xr.Dataset
    summary: dict[str, Any]


@dataclass(frozen=True)
class Stage4Fields:
    cells: pd.DataFrame
    dataset: xr.Dataset
    flux_levels: pd.DataFrame
    direction_comparison: pd.DataFrame
    persistence_review: pd.DataFrame
    low_alignment_review: pd.DataFrame
    representatives: pd.DataFrame
    summary: dict[str, Any]


def _append_flag(flags: np.ndarray, mask: np.ndarray, name: str) -> None:
    indexes = np.flatnonzero(mask)
    for index in indexes:
        flags[index] = f"{flags[index]};{name}" if flags[index] else name


def validate_transition_table(
    table: pd.DataFrame,
    grid: Any,
    config: ValidationConfig,
) -> ValidationResult:
    """Validate a supplied sparse table without changing its probabilities."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        error = f"missing_columns:{','.join(missing)}"
        return ValidationResult(
            links=table.copy(),
            rows=pd.DataFrame(),
            invalid_links=table.copy(),
            summary={"missing_columns": missing, "errors": [error]},
            errors=(error,),
        )

    links = table.loc[:, REQUIRED_COLUMNS].copy().reset_index(drop=True)
    links.insert(0, "transition_id", np.arange(len(links), dtype=np.int64))
    flags = np.full(len(links), "", dtype=object)
    errors: list[str] = []

    if links.empty:
        errors.append("empty_transition_table")

    for column in (*KEY_COLUMNS, "transition_count"):
        if not pd.api.types.is_integer_dtype(links[column]):
            errors.append(f"invalid_dtype:{column}")
    for column in (*REQUIRED_COLUMNS[4:8], "transition_probability"):
        if not pd.api.types.is_float_dtype(links[column]):
            errors.append(f"invalid_dtype:{column}")

    numeric_columns = REQUIRED_COLUMNS[4:]
    numeric_values = links.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric_values.to_numpy(dtype=float)).all(axis=1)
    _append_flag(flags, ~finite_mask, "non_finite")
    if not finite_mask.all():
        errors.append("non_finite_values")

    count_numeric = pd.to_numeric(links.transition_count, errors="coerce")
    count_valid = (
        count_numeric.notna() & np.isfinite(count_numeric) & (count_numeric > 0)
    )
    count_valid &= np.equal(count_numeric, np.floor(count_numeric))
    _append_flag(flags, ~count_valid.to_numpy(), "invalid_count")
    if not count_valid.all():
        errors.append("nonpositive_or_noninteger_transition_count")

    probability_numeric = pd.to_numeric(links.transition_probability, errors="coerce")
    probability_valid_range = probability_numeric.between(0.0, 1.0, inclusive="both")
    _append_flag(
        flags, ~probability_valid_range.fillna(False).to_numpy(), "probability_range"
    )
    if not probability_valid_range.fillna(False).all():
        errors.append("probability_out_of_range")

    duplicate_mask = links.duplicated(list(KEY_COLUMNS), keep=False).to_numpy()
    _append_flag(flags, duplicate_mask, "duplicate_key")
    if duplicate_mask.any():
        errors.append("duplicate_transition_keys")

    bins_numeric = links.loc[:, KEY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bin_valid = (
        bins_numeric.start_x_bin.between(0, grid.nx - 1)
        & bins_numeric.end_x_bin.between(0, grid.nx - 1)
        & bins_numeric.start_y_bin.between(0, grid.ny - 1)
        & bins_numeric.end_y_bin.between(0, grid.ny - 1)
    )
    _append_flag(flags, ~bin_valid.fillna(False).to_numpy(), "bin_bounds")
    if not bin_valid.fillna(False).all():
        errors.append("bin_out_of_bounds")

    expected_centers = {
        "start_x_center": grid.x_min
        + (bins_numeric.start_x_bin + 0.5) * grid.dx,
        "start_y_center": grid.y_min
        + (bins_numeric.start_y_bin + 0.5) * grid.dy,
        "end_x_center": grid.x_min + (bins_numeric.end_x_bin + 0.5) * grid.dx,
        "end_y_center": grid.y_min + (bins_numeric.end_y_bin + 0.5) * grid.dy,
    }
    center_valid = np.ones(len(links), dtype=bool)
    for column, expected in expected_centers.items():
        center_valid &= np.isclose(
            pd.to_numeric(links[column], errors="coerce").to_numpy(float),
            expected.to_numpy(float),
            rtol=0.0,
            atol=config.center_atol,
        )
    _append_flag(flags, ~center_valid, "grid_center")
    if not center_valid.all():
        errors.append("grid_center_mismatch")

    can_aggregate = (
        all(
            pd.api.types.is_integer_dtype(links[column])
            for column in (*KEY_COLUMNS, "transition_count")
        )
        and finite_mask.all()
    )
    rows = pd.DataFrame()
    probability_identity_valid = np.zeros(len(links), dtype=bool)
    if can_aggregate and len(links):
        links["start_cell_id"] = links.start_y_bin.astype(
            np.int64
        ) * grid.nx + links.start_x_bin.astype(np.int64)
        links["end_cell_id"] = links.end_y_bin.astype(
            np.int64
        ) * grid.nx + links.end_x_bin.astype(np.int64)
        links["is_stay"] = links.start_cell_id.eq(links.end_cell_id)
        grouped = links.groupby(
            ["start_cell_id", "start_x_bin", "start_y_bin"], sort=True
        )
        n_out = grouped.transition_count.transform("sum")
        links["N_out_total"] = n_out.astype(np.int64)
        links["expected_transition_probability"] = links.transition_count / n_out
        links["probability_residual"] = (
            links.transition_probability - links.expected_transition_probability
        )
        probability_identity_valid = np.isclose(
            links.transition_probability,
            links.expected_transition_probability,
            rtol=config.probability_rtol,
            atol=config.probability_atol,
        )
        _append_flag(flags, ~probability_identity_valid, "count_probability_identity")
        if not probability_identity_valid.all():
            errors.append("count_probability_identity_failure")

        rows = grouped.agg(
            N_out_total=("transition_count", "sum"),
            sum_probability=("transition_probability", "sum"),
            n_sparse_destinations=("transition_id", "size"),
        ).reset_index()
        rows["normalization_residual"] = rows.sum_probability - 1.0
        rows["normalization_valid"] = rows.normalization_residual.abs().le(
            config.normalization_atol
        )
        identity_by_cell = (
            pd.Series(probability_identity_valid, index=links.index)
            .groupby(links.start_cell_id)
            .all()
        )
        rows["count_probability_valid"] = rows.start_cell_id.map(identity_by_cell)
        rows["validation_flags"] = np.where(
            rows.normalization_valid & rows.count_probability_valid,
            "",
            "normalization_or_probability_identity",
        )
        invalid_cells = set(
            rows.loc[~rows.normalization_valid, "start_cell_id"].astype(int).tolist()
        )
        _append_flag(
            flags,
            links.start_cell_id.isin(invalid_cells).to_numpy(),
            "row_normalization",
        )
        if not rows.normalization_valid.all():
            errors.append("row_normalization_failure")

    links["validation_flags"] = flags
    invalid_links = links.loc[links.validation_flags.ne("")].copy()
    summary: dict[str, Any] = {
        "n_sparse_links": len(links),
        "total_transition_count": (
            int(links.transition_count.sum())
            if pd.api.types.is_integer_dtype(links.transition_count) and len(links)
            else None
        ),
        "populated_start_cells": len(rows),
        "duplicate_link_rows": int(duplicate_mask.sum()),
        "invalid_link_rows": len(invalid_links),
        "center_mismatch_rows": int((~center_valid).sum()),
        "normalization_failure_cells": (
            int((~rows.normalization_valid).sum()) if len(rows) else None
        ),
        "probability_identity_failure_rows": int((~probability_identity_valid).sum()),
        "normalization_residual_max_abs": (
            float(rows.normalization_residual.abs().max()) if len(rows) else None
        ),
        "probability_residual_max_abs": (
            float(links.probability_residual.abs().max())
            if "probability_residual" in links and len(links)
            else None
        ),
        "errors": errors,
    }
    return ValidationResult(links, rows, invalid_links, summary, tuple(errors))


def _support_dataset(cells: pd.DataFrame, grid: Any) -> xr.Dataset:
    coords = {
        "y": grid.y_min + (np.arange(grid.ny) + 0.5) * grid.dy,
        "x": grid.x_min + (np.arange(grid.nx) + 0.5) * grid.dx,
    }
    variables: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    excluded = {"cell_id", "x_bin", "y_bin", "x", "y"}
    for column in cells.columns:
        if column in excluded:
            continue
        series = cells[column]
        if not (
            pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        ):
            continue
        if pd.api.types.is_bool_dtype(series):
            array = np.zeros((grid.ny, grid.nx), dtype=np.int8)
            values = series.astype(np.int8).to_numpy()
        elif pd.api.types.is_integer_dtype(series):
            array = np.zeros((grid.ny, grid.nx), dtype=np.int64)
            values = series.to_numpy(dtype=np.int64)
        else:
            array = np.full((grid.ny, grid.nx), np.nan, dtype=float)
            values = series.to_numpy(dtype=float)
        array[cells.y_bin.to_numpy(dtype=int), cells.x_bin.to_numpy(dtype=int)] = (
            values
        )
        variables[column] = (("y", "x"), array)
    dataset = xr.Dataset(variables, coords=coords)
    dataset.attrs.update(
        representation="regular-grid Stage 0 support fields",
        x_periodic=str(grid.periodic_x).lower(),
        unpopulated_count_value=0,
        undefined_fraction_value="NaN",
    )
    return dataset


def compute_support_fields(
    links: pd.DataFrame,
    grid: Any,
    thresholds: tuple[int, ...],
) -> SupportResult:
    required = {"start_cell_id", "end_cell_id", "is_stay", "transition_count"}
    missing = sorted(required - set(links.columns))
    if missing:
        raise ValueError(f"Validated links missing support columns: {missing}")

    cell_ids = np.union1d(links.start_cell_id.unique(), links.end_cell_id.unique())
    cells = pd.DataFrame({"cell_id": cell_ids.astype(np.int64)})
    cells["x_bin"] = (cells.cell_id % grid.nx).astype(np.int64)
    cells["y_bin"] = (cells.cell_id // grid.nx).astype(np.int64)
    cells["x"] = grid.x_min + (cells.x_bin + 0.5) * grid.dx
    cells["y"] = grid.y_min + (cells.y_bin + 0.5) * grid.dy

    moving = links.loc[~links.is_stay]
    aggregations = {
        "N_out_total": links.groupby("start_cell_id").transition_count.sum(),
        "N_out_move": moving.groupby("start_cell_id").transition_count.sum(),
        "N_in_total": links.groupby("end_cell_id").transition_count.sum(),
        "N_in_move": moving.groupby("end_cell_id").transition_count.sum(),
        "C_stay": links.loc[links.is_stay]
        .groupby("start_cell_id")
        .transition_count.sum(),
        "n_distinct_moving_destinations": moving.groupby(
            "start_cell_id"
        ).end_cell_id.nunique(),
        "n_distinct_moving_sources": moving.groupby(
            "end_cell_id"
        ).start_cell_id.nunique(),
    }
    for column, values in aggregations.items():
        cells[column] = cells.cell_id.map(values).fillna(0).astype(np.int64)

    cells["P_stay"] = np.where(
        cells.N_out_total > 0, cells.C_stay / cells.N_out_total, np.nan
    )
    cells["P_move"] = np.where(cells.N_out_total > 0, 1.0 - cells.P_stay, np.nan)

    coverage_records: list[dict[str, Any]] = []
    domain_cells = grid.nx * grid.ny
    union_cells = len(cells)
    for count_field in COUNT_FIELDS:
        positive = int((cells[count_field] > 0).sum())
        for threshold in thresholds:
            flag_name = f"support_{count_field}_ge_{threshold}"
            cells[flag_name] = cells[count_field].ge(threshold)
            above = int(cells[flag_name].sum())
            coverage_records.append(
                {
                    "support_field": count_field,
                    "threshold": int(threshold),
                    "n_cells_above": above,
                    "n_positive_cells": positive,
                    "n_union_cells": union_cells,
                    "n_domain_cells": domain_cells,
                    "fraction_of_positive_cells": above / positive
                    if positive
                    else np.nan,
                    "fraction_of_union_cells": above / union_cells
                    if union_cells
                    else np.nan,
                    "fraction_of_domain_cells": above / domain_cells,
                }
            )

    coverage = pd.DataFrame(coverage_records)
    source = cells.loc[cells.N_out_total > 0]
    summary = {
        "union_populated_cells": int(union_cells),
        "populated_start_cells": int((cells.N_out_total > 0).sum()),
        "populated_destination_cells": int((cells.N_in_total > 0).sum()),
        "cells_with_outward_movement": int((cells.N_out_move > 0).sum()),
        "cells_with_inward_movement": int((cells.N_in_move > 0).sum()),
        "cells_with_stay_transitions": int((cells.C_stay > 0).sum()),
        "P_stay_mean": float(source.P_stay.mean()),
        "P_stay_median": float(source.P_stay.median()),
        "P_stay_q05": float(source.P_stay.quantile(0.05)),
        "P_stay_q95": float(source.P_stay.quantile(0.95)),
        "P_move_mean": float(source.P_move.mean()),
        "P_move_median": float(source.P_move.median()),
    }
    return SupportResult(
        cells=cells,
        dataset=_support_dataset(cells, grid),
        coverage=coverage,
        summary=summary,
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    total = float(sorted_weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.nan
    index = int(
        np.searchsorted(np.cumsum(sorted_weights), quantile * total, side="left")
    )
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _wrap_bearing_degrees(values: np.ndarray) -> np.ndarray:
    wrapped = np.remainder(np.asarray(values, dtype=float), 360.0)
    wrapped[wrapped >= 360.0] = 0.0
    wrapped[np.isclose(wrapped, 360.0, rtol=0.0, atol=1.0e-12)] = 0.0
    return wrapped


def _stage1_dataset(cells: pd.DataFrame, grid: Any) -> xr.Dataset:
    return _support_dataset(cells, grid).assign_attrs(
        representation="regular-grid Stage 0 support and Stage 1 outward fields",
        velocity_time_basis="configured timestep and time unit",
        direction_convention="degrees clockwise from positive y in [0, 360)",
    )


def compute_stage1_fields(
    links: pd.DataFrame,
    support_cells: pd.DataFrame,
    grid: Any,
    *,
    timestep: float,
    geometry: SpatialGeometry,
    config: Stage1Config,
) -> Stage1Fields:
    """Compute unmasked outward first moments from the supplied sparse links."""
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    required = {
        *REQUIRED_COLUMNS,
        "start_cell_id",
        "end_cell_id",
        "is_stay",
        "N_out_total",
    }
    missing = sorted(required - set(links.columns))
    if missing:
        raise ValueError(f"Validated links missing Stage 1 columns: {missing}")

    stage_links = links.copy()
    start_x = stage_links.start_x_center.to_numpy(float)
    start_y = stage_links.start_y_center.to_numpy(float)
    end_x = stage_links.end_x_center.to_numpy(float)
    end_y = stage_links.end_y_center.to_numpy(float)
    if len(stage_links) == 1:
        scalar_geometry = geometry.inverse(
            float(start_x[0]),
            float(start_y[0]),
            float(end_x[0]),
            float(end_y[0]),
        )
        forward, back, distance_length = (
            np.asarray([value], dtype=float) for value in scalar_geometry
        )
    else:
        forward, back, distance_length = geometry.inverse(
            start_x, start_y, end_x, end_y
        )
    distance_length = np.asarray(distance_length, dtype=float)
    moving_mask = ~stage_links.is_stay.to_numpy(bool)
    distance_length[~moving_mask] = 0.0
    source_bearing = _wrap_bearing_degrees(np.asarray(forward, dtype=float))
    source_side_bearing = _wrap_bearing_degrees(np.asarray(back, dtype=float))
    arrival_bearing = _wrap_bearing_degrees(source_side_bearing + 180.0)
    source_bearing[~moving_mask] = np.nan
    source_side_bearing[~moving_mask] = np.nan
    arrival_bearing[~moving_mask] = np.nan
    radians = np.deg2rad(np.nan_to_num(source_bearing, nan=0.0))

    stage_links["distance_length"] = distance_length
    stage_links["source_forward_bearing"] = source_bearing
    stage_links["theta_in_source"] = source_side_bearing
    stage_links["theta_in_motion_destination"] = arrival_bearing
    stage_links["dx_source_length"] = distance_length * np.sin(radians)
    stage_links["dy_source_length"] = distance_length * np.cos(radians)
    stage_links["conditional_moving_probability"] = np.nan
    n_move_by_source = (
        stage_links.loc[moving_mask].groupby("start_cell_id").transition_count.sum()
    )
    stage_links["N_out_move"] = (
        stage_links.start_cell_id.map(n_move_by_source).fillna(0).astype(np.int64)
    )
    stage_links.loc[moving_mask, "conditional_moving_probability"] = (
        stage_links.loc[moving_mask, "transition_count"]
        / stage_links.loc[moving_mask, "N_out_move"]
    )

    counts = stage_links.transition_count.to_numpy(float)
    stage_links["count_dx_length"] = counts * stage_links.dx_source_length
    stage_links["count_dy_length"] = counts * stage_links.dy_source_length
    stage_links["count_distance_length"] = counts * stage_links.distance_length
    all_moments = stage_links.groupby("start_cell_id", sort=True).agg(
        N_out_total_stage1=("transition_count", "sum"),
        sum_count_dx_all=("count_dx_length", "sum"),
        sum_count_dy_all=("count_dy_length", "sum"),
    )
    moving_links = stage_links.loc[moving_mask].copy()
    moving_moments = moving_links.groupby("start_cell_id", sort=True).agg(
        N_out_move_stage1=("transition_count", "sum"),
        sum_count_dx_move=("count_dx_length", "sum"),
        sum_count_dy_move=("count_dy_length", "sum"),
        sum_count_distance_move=("count_distance_length", "sum"),
    )
    moments = all_moments.join(moving_moments, how="left").reset_index()
    moments["N_out_move_stage1"] = moments.N_out_move_stage1.fillna(0).astype(np.int64)
    moments["mu_out_all_x_length"] = (
        moments.sum_count_dx_all / moments.N_out_total_stage1
    )
    moments["mu_out_all_y_length"] = (
        moments.sum_count_dy_all / moments.N_out_total_stage1
    )
    has_move = moments.N_out_move_stage1.gt(0)
    moments["mu_out_move_x_length"] = np.where(
        has_move,
        moments.sum_count_dx_move / moments.N_out_move_stage1,
        np.nan,
    )
    moments["mu_out_move_y_length"] = np.where(
        has_move,
        moments.sum_count_dy_move / moments.N_out_move_stage1,
        np.nan,
    )
    moments["mean_moving_distance_length"] = np.where(
        has_move,
        moments.sum_count_distance_move / moments.N_out_move_stage1,
        np.nan,
    )

    quantile_rows: list[dict[str, Any]] = []
    for cell_id, group in moving_links.groupby("start_cell_id", sort=True):
        record: dict[str, Any] = {"start_cell_id": int(cell_id)}
        for quantile in config.distance_quantiles:
            label = round(100 * quantile)
            record[f"moving_distance_q{label:02d}_length"] = _weighted_quantile(
                group.distance_length.to_numpy(float),
                group.transition_count.to_numpy(float),
                quantile,
            )
        quantile_rows.append(record)
    if quantile_rows:
        moments = moments.merge(
            pd.DataFrame(quantile_rows), on="start_cell_id", how="left"
        )

    vector_definitions = {
        "move": ("mu_out_move_x_length", "mu_out_move_y_length"),
        "all": ("mu_out_all_x_length", "mu_out_all_y_length"),
    }
    for label, (x_name, y_name) in vector_definitions.items():
        magnitude_name = f"mu_out_{label}_magnitude_length"
        moments[magnitude_name] = np.hypot(moments[x_name], moments[y_name])
        moments[f"U_out_{label}_x_rate"] = moments[x_name] / timestep
        moments[f"U_out_{label}_y_rate"] = moments[y_name] / timestep
        moments[f"U_out_{label}_magnitude_rate"] = (
            moments[magnitude_name] / timestep
        )
    move_magnitude = moments.mu_out_move_magnitude_length.to_numpy(float)
    theta_defined = np.isfinite(move_magnitude) & (
        move_magnitude > config.direction_zero_tolerance
    )
    theta = np.full(len(moments), np.nan, dtype=float)
    theta[theta_defined] = _wrap_bearing_degrees(
        np.rad2deg(
            np.arctan2(
                moments.loc[theta_defined, "mu_out_move_x_length"],
                moments.loc[theta_defined, "mu_out_move_y_length"],
            )
        )
    )
    moments["theta_mu_out"] = theta
    moments["theta_mu_out_defined"] = theta_defined

    stage_columns = [
        column
        for column in moments.columns
        if column
        not in {
            "sum_count_dx_all",
            "sum_count_dy_all",
            "sum_count_dx_move",
            "sum_count_dy_move",
            "sum_count_distance_move",
        }
    ]
    cells = support_cells.merge(
        moments.loc[:, stage_columns],
        left_on="cell_id",
        right_on="start_cell_id",
        how="left",
    ).drop(columns="start_cell_id")
    source_mask = cells.N_out_total.gt(0)
    cells["U_out_reduction_rate"] = (
        cells.U_out_move_magnitude_rate - cells.U_out_all_magnitude_rate
    )
    nonzero_move_vector = cells.mu_out_move_magnitude_length.gt(
        config.direction_zero_tolerance
    )
    cells["U_out_retained_fraction"] = np.where(
        nonzero_move_vector,
        cells.U_out_all_magnitude_rate / cells.U_out_move_magnitude_rate,
        np.nan,
    )
    cells["U_out_reduction_fraction"] = np.where(
        nonzero_move_vector, 1.0 - cells.U_out_retained_fraction, np.nan
    )
    cells["moment_identity_x_residual_length"] = (
        cells.mu_out_all_x_length - cells.P_move * cells.mu_out_move_x_length
    )
    cells["moment_identity_y_residual_length"] = (
        cells.mu_out_all_y_length - cells.P_move * cells.mu_out_move_y_length
    )
    cells["retained_fraction_minus_P_move"] = (
        cells.U_out_retained_fraction - cells.P_move
    )

    move_values = cells.loc[cells.N_out_move.gt(0), "U_out_move_magnitude_rate"]
    all_values = cells.loc[cells.N_out_move.gt(0), "U_out_all_magnitude_rate"]
    reduction_values = cells.loc[cells.N_out_move.gt(0), "U_out_reduction_rate"]
    strong_move_threshold = float(
        move_values.quantile(config.diagnostic_strong_flux_quantile)
    )
    strong_all_threshold = float(
        all_values.quantile(config.diagnostic_strong_flux_quantile)
    )
    reduction_threshold = float(
        reduction_values.quantile(config.diagnostic_reduction_quantile)
    )
    cells["diagnostic_strong_U_out_move"] = cells.U_out_move_magnitude_rate.ge(
        strong_move_threshold
    )
    cells["diagnostic_strong_U_out_all"] = cells.U_out_all_magnitude_rate.ge(
        strong_all_threshold
    )
    cells["diagnostic_top_reduction"] = cells.U_out_reduction_rate.ge(
        reduction_threshold
    )
    primary_count = config.primary_visualization_min_moving_count
    cells["diagnostic_weak_moving_support"] = cells.N_out_move.lt(primary_count)
    identity_x = cells.loc[source_mask, "moment_identity_x_residual_length"].abs()
    identity_y = cells.loc[source_mask, "moment_identity_y_residual_length"].abs()
    retained_residual = cells.loc[
        cells.U_out_retained_fraction.notna(), "retained_fraction_minus_P_move"
    ].abs()
    summary = {
        "cells_with_total_population_moment": int(source_mask.sum()),
        "cells_with_moving_conditioned_moment": int(cells.N_out_move.gt(0).sum()),
        "cells_with_theta_mu_out": int(cells.theta_mu_out.notna().sum()),
        "cells_with_zero_or_undefined_moving_vector": int(
            (cells.N_out_move.gt(0) & cells.theta_mu_out.isna()).sum()
        ),
        "moment_identity_max_abs_x_length": float(identity_x.max()),
        "moment_identity_max_abs_y_length": float(identity_y.max()),
        "retained_fraction_minus_P_move_max_abs": float(retained_residual.max()),
        "strong_flux_diagnostic_quantile": config.diagnostic_strong_flux_quantile,
        "strong_U_out_move_threshold_rate": strong_move_threshold,
        "strong_U_out_all_threshold_rate": strong_all_threshold,
        "top_reduction_diagnostic_quantile": config.diagnostic_reduction_quantile,
        "top_reduction_threshold_rate": reduction_threshold,
        "strong_U_out_move_cells_below_primary_support": int(
            (
                cells.diagnostic_strong_U_out_move
                & cells.diagnostic_weak_moving_support
            ).sum()
        ),
        "strong_U_out_all_cells_below_primary_support": int(
            (
                cells.diagnostic_strong_U_out_all & cells.diagnostic_weak_moving_support
            ).sum()
        ),
        "top_reduction_cells_below_primary_support": int(
            (
                cells.diagnostic_top_reduction & cells.diagnostic_weak_moving_support
            ).sum()
        ),
    }
    return Stage1Fields(
        links=stage_links,
        cells=cells,
        dataset=_stage1_dataset(cells, grid),
        summary=summary,
    )


def _circular_difference_degrees(
    first: np.ndarray | pd.Series, second: np.ndarray | pd.Series
) -> np.ndarray:
    return np.abs(signed_angle_difference(first, second))


def compute_stage2_fields(
    links: pd.DataFrame,
    stage1_cells: pd.DataFrame,
    grid: GridConfig,
    *,
    stage1: Stage1Config,
    config: Stage2Config,
) -> Stage2Fields:
    """Compute unmasked moving-only circular diagnostics for Stage 2."""
    required_links = {
        "start_cell_id",
        "is_stay",
        "transition_count",
        "conditional_moving_probability",
        "source_forward_bearing",
        "distance_length",
    }
    missing_links = sorted(required_links - set(links.columns))
    if missing_links:
        raise ValueError(f"Stage 1 links missing Stage 2 columns: {missing_links}")
    required_cells = {
        "cell_id",
        "N_out_move",
        "theta_mu_out",
        "U_out_all_magnitude_rate",
        "diagnostic_strong_U_out_all",
        "diagnostic_strong_U_out_move",
    }
    missing_cells = sorted(required_cells - set(stage1_cells.columns))
    if missing_cells:
        raise ValueError(f"Stage 1 cells missing Stage 2 columns: {missing_cells}")

    moving = links.loc[~links.is_stay].copy()
    if moving.empty:
        raise ValueError("Stage 2 requires at least one moving transition")
    probability = moving.conditional_moving_probability.to_numpy(float)
    angle = np.deg2rad(moving.source_forward_bearing.to_numpy(float))
    moving["M1_real_contribution"] = probability * np.cos(angle)
    moving["M1_imag_contribution"] = probability * np.sin(angle)
    moving["M2_real_contribution"] = probability * np.cos(2.0 * angle)
    moving["M2_imag_contribution"] = probability * np.sin(2.0 * angle)
    moving["count_distance_length_stage2"] = moving.transition_count * moving.distance_length
    moving["count_distance_squared_length2"] = (
        moving.transition_count * moving.distance_length**2
    )
    harmonics = (
        moving.groupby("start_cell_id", sort=True)
        .agg(
            M1_out_real=("M1_real_contribution", "sum"),
            M1_out_imag=("M1_imag_contribution", "sum"),
            M2_out_real=("M2_real_contribution", "sum"),
            M2_out_imag=("M2_imag_contribution", "sum"),
            N_out_move_stage2=("transition_count", "sum"),
            moving_distance_max_length=("distance_length", "max"),
            sum_count_distance_length=("count_distance_length_stage2", "sum"),
            sum_count_distance_squared_length2=("count_distance_squared_length2", "sum"),
            max_count_distance_length=("count_distance_length_stage2", "max"),
        )
        .reset_index()
    )
    harmonics["R1_out"] = np.clip(
        np.hypot(harmonics.M1_out_real, harmonics.M1_out_imag), 0.0, 1.0
    )
    harmonics["R2_out"] = np.clip(
        np.hypot(harmonics.M2_out_real, harmonics.M2_out_imag), 0.0, 1.0
    )
    r1_defined = harmonics.R1_out.gt(config.harmonic_zero_tolerance)
    r2_defined = harmonics.R2_out.gt(config.harmonic_zero_tolerance)
    harmonics["theta1_out"] = np.nan
    harmonics.loc[r1_defined, "theta1_out"] = _wrap_bearing_degrees(
        np.rad2deg(
            np.arctan2(
                harmonics.loc[r1_defined, "M1_out_imag"],
                harmonics.loc[r1_defined, "M1_out_real"],
            )
        )
    )
    harmonics["theta2_out"] = np.nan
    theta2_raw = 0.5 * np.rad2deg(
        np.arctan2(
            harmonics.loc[r2_defined, "M2_out_imag"],
            harmonics.loc[r2_defined, "M2_out_real"],
        )
    )
    harmonics.loc[r2_defined, "theta2_out"] = np.remainder(theta2_raw, 180.0)

    bin_width = 360.0 / config.angular_bins
    moving["angular_bin"] = (
        np.floor(moving.source_forward_bearing / bin_width)
        .astype(int)
        .clip(0, config.angular_bins - 1)
    )
    bin_mass = (
        moving.groupby(["start_cell_id", "angular_bin"], sort=True)
        .conditional_moving_probability.sum()
        .reset_index(name="angular_bin_probability")
    )
    bin_mass["entropy_contribution"] = -bin_mass.angular_bin_probability * np.log(
        bin_mass.angular_bin_probability
    )
    entropy = (
        bin_mass.groupby("start_cell_id", sort=True)
        .agg(
            angular_entropy_numerator=("entropy_contribution", "sum"),
            n_occupied_angular_bins=("angular_bin", "nunique"),
        )
        .reset_index()
    )
    entropy["angular_entropy_out"] = entropy.angular_entropy_numerator / np.log(
        config.angular_bins
    )
    entropy["effective_angular_bins"] = np.exp(entropy.angular_entropy_numerator)
    harmonics = harmonics.merge(entropy, on="start_cell_id", how="left")
    harmonics["mean_moving_distance_stage2_length"] = (
        harmonics.sum_count_distance_length / harmonics.N_out_move_stage2
    )
    second_moment = (
        harmonics.sum_count_distance_squared_length2 / harmonics.N_out_move_stage2
    )
    harmonics["moving_distance_std_length"] = np.sqrt(
        np.maximum(
            second_moment - harmonics.mean_moving_distance_stage2_length**2,
            0.0,
        )
    )
    harmonics["max_to_mean_moving_distance_ratio"] = (
        harmonics.moving_distance_max_length / harmonics.mean_moving_distance_stage2_length
    )
    harmonics["longest_link_distance_leverage_fraction"] = (
        harmonics.max_count_distance_length / harmonics.sum_count_distance_length
    )

    retained_harmonics = [
        column
        for column in harmonics.columns
        if column
        not in {
            "sum_count_distance_length",
            "sum_count_distance_squared_length2",
            "max_count_distance_length",
            "angular_entropy_numerator",
        }
    ]
    cells = stage1_cells.merge(
        harmonics.loc[:, retained_harmonics],
        left_on="cell_id",
        right_on="start_cell_id",
        how="left",
    ).drop(columns="start_cell_id")
    moving_count_check = cells.loc[
        cells.N_out_move.gt(0), ["N_out_move", "N_out_move_stage2"]
    ]
    if not np.array_equal(
        moving_count_check.N_out_move.to_numpy(),
        moving_count_check.N_out_move_stage2.to_numpy(),
    ):
        raise ValueError("Stage 2 moving counts do not match the Stage 1 fields")
    cells["delta_theta_mu1_out"] = np.nan
    delta_defined = cells.theta_mu_out.notna() & cells.theta1_out.notna()
    cells.loc[delta_defined, "delta_theta_mu1_out"] = _circular_difference_degrees(
        cells.loc[delta_defined, "theta_mu_out"],
        cells.loc[delta_defined, "theta1_out"],
    )
    cells["R1_out_defined"] = cells.R1_out.notna()
    cells["theta1_out_defined"] = cells.theta1_out.notna()
    cells["theta2_out_defined"] = cells.theta2_out.notna()

    primary = cells.N_out_move.ge(stage1.primary_visualization_min_moving_count)
    sensitivity = cells.N_out_move.ge(stage1.sensitivity_visualization_min_moving_count)
    strong = cells.diagnostic_strong_U_out_all & primary
    primary_fields = cells.loc[
        primary,
        [
            "U_out_all_magnitude_rate",
            "R1_out",
            "R2_out",
            "angular_entropy_out",
            "delta_theta_mu1_out",
            "max_to_mean_moving_distance_ratio",
            "longest_link_distance_leverage_fraction",
        ],
    ]
    correlations = primary_fields.corr(method="spearman")
    summary = {
        "angular_bins": config.angular_bins,
        "angular_bin_width_degrees": bin_width,
        "cells_with_R1_out": int(cells.R1_out.notna().sum()),
        "cells_with_theta1_out": int(cells.theta1_out.notna().sum()),
        "cells_with_R2_out": int(cells.R2_out.notna().sum()),
        "cells_with_theta2_out": int(cells.theta2_out.notna().sum()),
        "cells_with_entropy": int(cells.angular_entropy_out.notna().sum()),
        "cells_with_delta_theta_mu1_out": int(cells.delta_theta_mu1_out.notna().sum()),
        "primary_visualization_cells": int(primary.sum()),
        "sensitivity_visualization_cells": int(sensitivity.sum()),
        "strong_flux_primary_cells": int(strong.sum()),
        "strong_flux_high_R1_cells": int(
            (strong & cells.R1_out.ge(config.high_R1)).sum()
        ),
        "strong_flux_below_high_R1_cells": int(
            (strong & cells.R1_out.lt(config.high_R1)).sum()
        ),
        "strong_flux_large_delta_cells": int(
            (
                strong & cells.delta_theta_mu1_out.ge(config.large_delta_theta_degrees)
            ).sum()
        ),
        "high_R2_low_R1_primary_cells": int(
            (
                primary
                & cells.R2_out.ge(config.high_R2)
                & cells.R1_out.le(config.low_R1)
            ).sum()
        ),
        "candidate_split_primary_cells": int(
            (
                primary
                & cells.R1_out.between(config.split_R1_min, config.split_R1_max)
                & cells.R2_out.le(config.split_R2_max)
            ).sum()
        ),
        "spearman_U_out_all_vs_R1_primary": float(
            correlations.loc["U_out_all_magnitude_rate", "R1_out"]
        ),
        "spearman_delta_vs_max_mean_distance_ratio_primary": float(
            correlations.loc["delta_theta_mu1_out", "max_to_mean_moving_distance_ratio"]
        ),
        "spearman_delta_vs_longest_link_leverage_primary": float(
            correlations.loc[
                "delta_theta_mu1_out", "longest_link_distance_leverage_fraction"
            ]
        ),
        "optional_angular_peak_diagnostic_implemented": False,
        "optional_angular_peak_diagnostic_decision": (
            "not implemented because R1/R2/delta_theta plus retained raw "
            "representative angular distributions resolve the important cases"
        ),
    }
    return Stage2Fields(
        cells=cells,
        dataset=_stage1_dataset(cells, grid).assign_attrs(
            stage2_angular_bins=config.angular_bins,
            stage2_angular_bin_width_degrees=bin_width,
            stage2_peak_diagnostic=(
                "not implemented: circular diagnostics and retained raw representative "
                "angular distributions resolve the important cases"
            ),
        ),
        summary=summary,
    )


def compute_stage3_fields(
    links: pd.DataFrame,
    stage2_cells: pd.DataFrame,
    grid: GridConfig,
    *,
    stage1: Stage1Config,
    stage2: Stage2Config,
    config: Stage3Config,
) -> Stage3Fields:
    """Compute unmasked destination-conditioned incoming and alignment diagnostics."""
    required_links = {
        "start_cell_id",
        "end_cell_id",
        "is_stay",
        "transition_count",
        "distance_length",
        "source_forward_bearing",
        "theta_in_source",
        "theta_in_motion_destination",
        "conditional_moving_probability",
    }
    missing_links = sorted(required_links - set(links.columns))
    if missing_links:
        raise ValueError(f"Stage 1 links missing Stage 3 columns: {missing_links}")
    required_cells = {
        "cell_id",
        "N_out_move",
        "N_in_move",
        "U_out_all_magnitude_rate",
        "diagnostic_strong_U_out_all",
        "theta_mu_out",
        "R1_out",
        "theta1_out",
        "R2_out",
        "angular_entropy_out",
        "delta_theta_mu1_out",
    }
    missing_cells = sorted(required_cells - set(stage2_cells.columns))
    if missing_cells:
        raise ValueError(f"Stage 2 cells missing Stage 3 columns: {missing_cells}")

    moving = links.loc[~links.is_stay].copy()
    if moving.empty:
        raise ValueError("Stage 3 requires at least one moving transition")
    incoming_counts = moving.groupby("end_cell_id").transition_count.transform("sum")
    moving["N_in_move_stage3"] = incoming_counts.astype(np.int64)
    moving["conditional_incoming_probability"] = (
        moving.transition_count / moving.N_in_move_stage3
    )
    probability = moving.conditional_incoming_probability.to_numpy(float)
    source_angle = np.deg2rad(moving.theta_in_source.to_numpy(float))
    motion_angle = np.deg2rad(moving.theta_in_motion_destination.to_numpy(float))
    moving["M1_in_source_real_contribution"] = probability * np.cos(source_angle)
    moving["M1_in_source_imag_contribution"] = probability * np.sin(source_angle)
    moving["M1_in_motion_real_contribution"] = probability * np.cos(motion_angle)
    moving["M1_in_motion_imag_contribution"] = probability * np.sin(motion_angle)
    moving["M2_in_real_contribution"] = probability * np.cos(2.0 * motion_angle)
    moving["M2_in_imag_contribution"] = probability * np.sin(2.0 * motion_angle)
    moving["count_in_x_length"] = (
        moving.transition_count * moving.distance_length * np.sin(motion_angle)
    )
    moving["count_in_y_length"] = (
        moving.transition_count * moving.distance_length * np.cos(motion_angle)
    )
    moving["count_in_distance_length"] = moving.transition_count * moving.distance_length

    incoming = (
        moving.groupby("end_cell_id", sort=True)
        .agg(
            M1_in_source_real=("M1_in_source_real_contribution", "sum"),
            M1_in_source_imag=("M1_in_source_imag_contribution", "sum"),
            M1_in_motion_real=("M1_in_motion_real_contribution", "sum"),
            M1_in_motion_imag=("M1_in_motion_imag_contribution", "sum"),
            M2_in_real=("M2_in_real_contribution", "sum"),
            M2_in_imag=("M2_in_imag_contribution", "sum"),
            N_in_move_stage3=("transition_count", "sum"),
            sum_count_in_x_length=("count_in_x_length", "sum"),
            sum_count_in_y_length=("count_in_y_length", "sum"),
            sum_count_in_distance_length=("count_in_distance_length", "sum"),
            incoming_moving_distance_max_length=("distance_length", "max"),
        )
        .reset_index()
    )
    r1_source = np.hypot(incoming.M1_in_source_real, incoming.M1_in_source_imag)
    incoming["R1_in"] = np.clip(
        np.hypot(incoming.M1_in_motion_real, incoming.M1_in_motion_imag), 0.0, 1.0
    )
    if not np.allclose(r1_source, incoming.R1_in, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Incoming source-side and arrival-motion R1 magnitudes disagree"
        )
    incoming["R2_in"] = np.clip(
        np.hypot(incoming.M2_in_real, incoming.M2_in_imag), 0.0, 1.0
    )
    r1_defined = incoming.R1_in.gt(config.harmonic_zero_tolerance)
    r2_defined = incoming.R2_in.gt(config.harmonic_zero_tolerance)
    incoming["theta1_in_source"] = np.nan
    incoming.loc[r1_defined, "theta1_in_source"] = _wrap_bearing_degrees(
        np.rad2deg(
            np.arctan2(
                incoming.loc[r1_defined, "M1_in_source_imag"],
                incoming.loc[r1_defined, "M1_in_source_real"],
            )
        )
    )
    incoming["theta1_in_motion_destination"] = np.nan
    incoming.loc[r1_defined, "theta1_in_motion_destination"] = _wrap_bearing_degrees(
        np.rad2deg(
            np.arctan2(
                incoming.loc[r1_defined, "M1_in_motion_imag"],
                incoming.loc[r1_defined, "M1_in_motion_real"],
            )
        )
    )
    incoming["theta2_in"] = np.nan
    theta2_raw = 0.5 * np.rad2deg(
        np.arctan2(
            incoming.loc[r2_defined, "M2_in_imag"],
            incoming.loc[r2_defined, "M2_in_real"],
        )
    )
    incoming.loc[r2_defined, "theta2_in"] = np.remainder(theta2_raw, 180.0)

    bin_width = 360.0 / config.angular_bins
    moving["incoming_angular_bin"] = (
        np.floor(moving.theta_in_motion_destination / bin_width)
        .astype(int)
        .clip(0, config.angular_bins - 1)
    )
    bin_mass = (
        moving.groupby(["end_cell_id", "incoming_angular_bin"], sort=True)
        .conditional_incoming_probability.sum()
        .reset_index(name="incoming_angular_bin_probability")
    )
    bin_mass["entropy_contribution"] = (
        -bin_mass.incoming_angular_bin_probability
        * np.log(bin_mass.incoming_angular_bin_probability)
    )
    entropy = (
        bin_mass.groupby("end_cell_id", sort=True)
        .agg(
            incoming_angular_entropy_numerator=("entropy_contribution", "sum"),
            n_occupied_incoming_angular_bins=("incoming_angular_bin", "nunique"),
        )
        .reset_index()
    )
    entropy["H_in"] = entropy.incoming_angular_entropy_numerator / np.log(
        config.angular_bins
    )
    entropy["effective_incoming_angular_bins"] = np.exp(
        entropy.incoming_angular_entropy_numerator
    )
    incoming = incoming.merge(entropy, on="end_cell_id", how="left")

    incoming["mu_in_move_x_length"] = (
        incoming.sum_count_in_x_length / incoming.N_in_move_stage3
    )
    incoming["mu_in_move_y_length"] = (
        incoming.sum_count_in_y_length / incoming.N_in_move_stage3
    )
    incoming["mu_in_move_magnitude_length"] = np.hypot(
        incoming.mu_in_move_x_length, incoming.mu_in_move_y_length
    )
    incoming["incoming_moving_distance_mean_length"] = (
        incoming.sum_count_in_distance_length / incoming.N_in_move_stage3
    )
    mu_defined = incoming.mu_in_move_magnitude_length.gt(config.harmonic_zero_tolerance)
    incoming["theta_mu_in_motion_destination"] = np.nan
    incoming.loc[mu_defined, "theta_mu_in_motion_destination"] = _wrap_bearing_degrees(
        np.rad2deg(
            np.arctan2(
                incoming.loc[mu_defined, "mu_in_move_x_length"],
                incoming.loc[mu_defined, "mu_in_move_y_length"],
            )
        )
    )
    incoming["delta_theta_mu1_in"] = np.nan
    delta_defined = (
        incoming.theta_mu_in_motion_destination.notna()
        & incoming.theta1_in_motion_destination.notna()
    )
    incoming.loc[delta_defined, "delta_theta_mu1_in"] = _circular_difference_degrees(
        incoming.loc[delta_defined, "theta_mu_in_motion_destination"],
        incoming.loc[delta_defined, "theta1_in_motion_destination"],
    )
    retained = [
        column
        for column in incoming.columns
        if column
        not in {
            "sum_count_in_x_length",
            "sum_count_in_y_length",
            "sum_count_in_distance_length",
            "incoming_angular_entropy_numerator",
        }
    ]
    cells = stage2_cells.merge(
        incoming.loc[:, retained],
        left_on="cell_id",
        right_on="end_cell_id",
        how="left",
    ).drop(columns="end_cell_id")
    count_check = cells.loc[cells.N_in_move.gt(0), ["N_in_move", "N_in_move_stage3"]]
    if not np.array_equal(
        count_check.N_in_move.to_numpy(), count_check.N_in_move_stage3.to_numpy()
    ):
        raise ValueError("Stage 3 incoming counts do not match the support fields")
    cells["H_out"] = cells.angular_entropy_out
    cells["A_io"] = np.nan
    aio_defined = cells.theta1_out.notna() & cells.theta1_in_motion_destination.notna()
    cells.loc[aio_defined, "A_io"] = np.cos(
        np.deg2rad(
            cells.loc[aio_defined, "theta1_out"]
            - cells.loc[aio_defined, "theta1_in_motion_destination"]
        )
    )
    cells["A_io_mu"] = np.nan
    aio_mu_defined = (
        cells.theta_mu_out.notna() & cells.theta_mu_in_motion_destination.notna()
    )
    cells.loc[aio_mu_defined, "A_io_mu"] = np.cos(
        np.deg2rad(
            cells.loc[aio_mu_defined, "theta_mu_out"]
            - cells.loc[aio_mu_defined, "theta_mu_in_motion_destination"]
        )
    )
    cells["A_io_minus_A_io_mu"] = cells.A_io - cells.A_io_mu
    cells["abs_A_io_minus_A_io_mu"] = cells.A_io_minus_A_io_mu.abs()
    primary = stage1.primary_visualization_min_moving_count
    sensitivity = stage1.sensitivity_visualization_min_moving_count
    cells["stage3_incoming_primary_mask"] = cells.N_in_move.ge(primary)
    cells["stage3_incoming_sensitivity_mask"] = cells.N_in_move.ge(sensitivity)
    cells["stage3_joint_primary_mask"] = cells.N_out_move.ge(
        primary
    ) & cells.N_in_move.ge(primary)
    cells["stage3_joint_sensitivity_mask"] = cells.N_out_move.ge(
        sensitivity
    ) & cells.N_in_move.ge(sensitivity)

    joint_primary = cells.stage3_joint_primary_mask
    joint_sensitivity = cells.stage3_joint_sensitivity_mask
    strong_coherent_out_primary = (
        cells.diagnostic_strong_U_out_all
        & cells.R1_out.ge(stage2.high_R1)
        & joint_primary
    )
    strong_coherent_out_sensitivity = (
        cells.diagnostic_strong_U_out_all
        & cells.R1_out.ge(stage2.high_R1)
        & joint_sensitivity
    )
    coherent_both_primary = strong_coherent_out_primary & cells.R1_in.ge(stage2.high_R1)
    alignment_pairs = cells.loc[
        joint_primary & cells.A_io.notna() & cells.A_io_mu.notna(),
        ["A_io", "A_io_mu"],
    ]
    alignment_spearman = alignment_pairs.corr(method="spearman").iloc[0, 1]
    summary = {
        "angular_bins": config.angular_bins,
        "angular_bin_width_degrees": bin_width,
        "cells_with_R1_in": int(cells.R1_in.notna().sum()),
        "cells_with_theta1_in_source": int(cells.theta1_in_source.notna().sum()),
        "cells_with_theta1_in_motion_destination": int(
            cells.theta1_in_motion_destination.notna().sum()
        ),
        "cells_with_R2_in": int(cells.R2_in.notna().sum()),
        "cells_with_theta2_in": int(cells.theta2_in.notna().sum()),
        "cells_with_H_in": int(cells.H_in.notna().sum()),
        "cells_with_theta_mu_in_motion_destination": int(
            cells.theta_mu_in_motion_destination.notna().sum()
        ),
        "cells_with_delta_theta_mu1_in": int(cells.delta_theta_mu1_in.notna().sum()),
        "cells_with_A_io": int(cells.A_io.notna().sum()),
        "cells_with_A_io_mu": int(cells.A_io_mu.notna().sum()),
        "incoming_primary_mask_cells": int(cells.stage3_incoming_primary_mask.sum()),
        "incoming_sensitivity_mask_cells": int(
            cells.stage3_incoming_sensitivity_mask.sum()
        ),
        "joint_primary_mask_cells": int(joint_primary.sum()),
        "joint_sensitivity_mask_cells": int(joint_sensitivity.sum()),
        "strong_high_R1_out_joint_primary_cells": int(
            strong_coherent_out_primary.sum()
        ),
        "strong_high_R1_out_high_R1_in_joint_primary_cells": int(
            coherent_both_primary.sum()
        ),
        "strong_high_R1_out_joint_sensitivity_cells": int(
            strong_coherent_out_sensitivity.sum()
        ),
        "strong_high_R1_out_high_R1_in_joint_sensitivity_cells": int(
            (strong_coherent_out_sensitivity & cells.R1_in.ge(stage2.high_R1)).sum()
        ),
        "strong_high_R1_out_low_R1_in_joint_primary_cells": int(
            (strong_coherent_out_primary & cells.R1_in.le(stage2.low_R1)).sum()
        ),
        "strong_coherent_in_and_out_poor_A_io_joint_primary_cells": int(
            (
                coherent_both_primary
                & cells.A_io.le(config.diagnostic_poor_alignment_max)
            ).sum()
        ),
        "coherent_in_and_out_poor_A_io_joint_primary_cells": int(
            (
                joint_primary
                & cells.R1_out.ge(stage2.high_R1)
                & cells.R1_in.ge(stage2.high_R1)
                & cells.A_io.le(config.diagnostic_poor_alignment_max)
            ).sum()
        ),
        "coherent_in_and_out_poor_A_io_joint_sensitivity_cells": int(
            (
                joint_sensitivity
                & cells.R1_out.ge(stage2.high_R1)
                & cells.R1_in.ge(stage2.high_R1)
                & cells.A_io.le(config.diagnostic_poor_alignment_max)
            ).sum()
        ),
        "spearman_A_io_vs_A_io_mu_joint_primary": float(alignment_spearman),
        "median_abs_A_io_minus_A_io_mu_joint_primary": float(
            alignment_pairs.A_io.sub(alignment_pairs.A_io_mu).abs().median()
        ),
        "q95_abs_A_io_minus_A_io_mu_joint_primary": float(
            alignment_pairs.A_io.sub(alignment_pairs.A_io_mu).abs().quantile(0.95)
        ),
        "neighborhood_persistence_implemented": False,
        "geographic_or_current_specific_filters_applied": False,
    }
    dataset = _stage1_dataset(cells, grid).assign_attrs(
        stage3_angular_bins=config.angular_bins,
        stage3_angular_bin_width_degrees=bin_width,
        incoming_conditioning="transition_count / destination N_in_move",
        incoming_motion_angle="destination-local arrival-motion bearing",
        alignment_primary="cos(theta1_out - theta1_in_motion_destination)",
        alignment_secondary=("cos(theta_mu_out - theta_mu_in_motion_destination)"),
        stage3b_neighborhood_persistence="not implemented",
        geographic_filters="none",
    )
    stage3_links = links.copy()
    stage3_links["conditional_incoming_probability"] = np.nan
    stage3_links.loc[moving.index, "conditional_incoming_probability"] = (
        moving.conditional_incoming_probability
    )
    stage3_links["N_in_move_stage3"] = 0
    stage3_links.loc[moving.index, "N_in_move_stage3"] = moving.N_in_move_stage3
    stage3_links["N_in_move_stage3"] = stage3_links.N_in_move_stage3.astype(np.int64)
    return Stage3Fields(
        links=stage3_links,
        cells=cells,
        dataset=dataset,
        summary=summary,
    )


def _neighbor_directional_consistency(
    cells: pd.DataFrame,
    grid: GridConfig,
    *,
    direction_field: str,
    support_field: str,
    support_threshold: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform 8-neighbor cosine consistency without bridging absent cells."""
    required = {"cell_id", "x_bin", "y_bin", direction_field, support_field}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Neighborhood calculation missing columns: {missing}")
    if cells.cell_id.duplicated().any():
        raise ValueError("Neighborhood calculation requires unique cell_id values")
    direction_by_id = cells.set_index("cell_id")[direction_field]
    support_by_id = cells.set_index("cell_id")[support_field]
    focal_direction = cells[direction_field].to_numpy(float)
    focal_support = cells[support_field].to_numpy(float)
    x_bin = cells.x_bin.to_numpy(np.int64)
    y_bin = cells.y_bin.to_numpy(np.int64)
    contribution_sum = np.zeros(len(cells), dtype=float)
    valid_neighbor_count = np.zeros(len(cells), dtype=np.int64)
    for delta_lat, delta_lon in NEIGHBOR_OFFSETS_8:
        neighbor_lat = y_bin + delta_lat
        neighbor_lon = x_bin + delta_lon
        valid_coordinate = (neighbor_lat >= 0) & (neighbor_lat < grid.ny)
        if grid.periodic_x:
            neighbor_lon = np.remainder(neighbor_lon, grid.nx)
        else:
            valid_coordinate &= (neighbor_lon >= 0) & (neighbor_lon < grid.nx)
        neighbor_id = neighbor_lat * grid.nx + neighbor_lon
        mapped_id = pd.Series(neighbor_id).where(valid_coordinate)
        neighbor_direction = mapped_id.map(direction_by_id).to_numpy(float)
        neighbor_support = mapped_id.map(support_by_id).to_numpy(float)
        valid = (
            valid_coordinate
            & np.isfinite(focal_direction)
            & np.isfinite(neighbor_direction)
        )
        if support_threshold is not None:
            valid &= (focal_support >= support_threshold) & (
                neighbor_support >= support_threshold
            )
        contribution = np.cos(np.deg2rad(focal_direction - neighbor_direction))
        contribution_sum[valid] += contribution[valid]
        valid_neighbor_count[valid] += 1
    consistency = np.full(len(cells), np.nan, dtype=float)
    defined = valid_neighbor_count > 0
    consistency[defined] = contribution_sum[defined] / valid_neighbor_count[defined]
    return consistency, valid_neighbor_count


def _signed_circular_difference_degrees(
    first: np.ndarray | pd.Series, second: np.ndarray | pd.Series
) -> np.ndarray:
    return signed_angle_difference(first, second)


def _stage4_flux_level_comparison(
    cells: pd.DataFrame,
    stage1: Stage1Config,
    config: Stage4Config,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for support_threshold in (
        stage1.primary_visualization_min_moving_count,
        stage1.sensitivity_visualization_min_moving_count,
    ):
        population = (
            cells.N_out_move.ge(support_threshold)
            & cells.U_out_all_magnitude_rate.notna()
            & cells.U_coh_rate.notna()
        )
        values = cells.loc[population, ["U_out_all_magnitude_rate", "U_coh_rate"]]
        for quantile in config.flux_percentiles:
            label = round(100 * quantile)
            threshold_u = float(values.U_out_all_magnitude_rate.quantile(quantile))
            threshold_coh = float(values.U_coh_rate.quantile(quantile))
            selected_u = population & cells.U_out_all_magnitude_rate.ge(threshold_u)
            selected_coh = population & cells.U_coh_rate.ge(threshold_coh)
            intersection = selected_u & selected_coh
            union = selected_u | selected_coh
            cells[f"stage4_U_out_all_ge_q{label}_support_{support_threshold}"] = (
                selected_u
            )
            cells[f"stage4_U_coh_ge_q{label}_support_{support_threshold}"] = (
                selected_coh
            )
            records.append(
                {
                    "support_threshold": support_threshold,
                    "quantile": quantile,
                    "U_out_all_threshold_rate": threshold_u,
                    "U_coh_threshold_rate": threshold_coh,
                    "n_population": int(population.sum()),
                    "n_U_out_all_selected": int(selected_u.sum()),
                    "n_U_coh_selected": int(selected_coh.sum()),
                    "n_intersection": int(intersection.sum()),
                    "n_union": int(union.sum()),
                    "jaccard_overlap": float(intersection.sum() / union.sum()),
                    "fraction_U_out_all_retained_by_U_coh": float(
                        intersection.sum() / selected_u.sum()
                    ),
                    "fraction_U_coh_also_selected_by_U_out_all": float(
                        intersection.sum() / selected_coh.sum()
                    ),
                    "median_R1_out_U_out_all_selected": float(
                        cells.loc[selected_u, "R1_out"].median()
                    ),
                    "median_R1_out_U_out_all_removed_by_U_coh": float(
                        cells.loc[selected_u & ~selected_coh, "R1_out"].median()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _stage4_direction_comparison(
    cells: pd.DataFrame,
    stage1: Stage1Config,
    stage2: Stage2Config,
    config: Stage4Config,
) -> pd.DataFrame:
    primary = stage1.primary_visualization_min_moving_count
    sensitivity = stage1.sensitivity_visualization_min_moving_count
    populations = {
        f"outgoing_ge_{primary}": cells.N_out_move.ge(primary),
        f"outgoing_ge_{sensitivity}": cells.N_out_move.ge(sensitivity),
        "strong_high_R1_out_primary": (
            cells.diagnostic_strong_U_out_all
            & cells.R1_out.ge(stage2.high_R1)
            & cells.N_out_move.ge(primary)
        ),
        "strong_high_R1_out_sensitivity": (
            cells.diagnostic_strong_U_out_all
            & cells.R1_out.ge(stage2.high_R1)
            & cells.N_out_move.ge(sensitivity)
        ),
        "strong_reliable_through_primary": (
            cells.diagnostic_strong_U_out_all
            & cells.R1_out.ge(stage2.high_R1)
            & cells.R1_in.ge(stage2.high_R1)
            & cells.N_out_move.ge(primary)
            & cells.N_in_move.ge(primary)
        ),
        "strong_reliable_through_sensitivity": (
            cells.diagnostic_strong_U_out_all
            & cells.R1_out.ge(stage2.high_R1)
            & cells.R1_in.ge(stage2.high_R1)
            & cells.N_out_move.ge(sensitivity)
            & cells.N_in_move.ge(sensitivity)
        ),
    }
    fields = (
        "delta_theta_mu1_out",
        "delta_theta_mu1_in",
        "abs_delta_theta_io_1",
        "abs_delta_theta_io_mu",
        "abs_delta_theta_io_1_minus_mu",
    )
    records: list[dict[str, Any]] = []
    for population_name, mask in populations.items():
        for field_name in fields:
            values = cells.loc[mask & cells[field_name].notna(), field_name]
            record: dict[str, Any] = {
                "population": population_name,
                "field": field_name,
                "n_cells": len(values),
                "n_ge_direction_review_degrees": int(
                    values.ge(config.direction_difference_review_degrees).sum()
                ),
                "n_ge_substantial_turn_degrees": int(
                    values.ge(config.substantial_turn_degrees).sum()
                ),
            }
            for quantile in (0.0, 0.05, 0.5, 0.95, 0.99, 1.0):
                record[f"q{quantile:g}"] = (
                    float(values.quantile(quantile)) if len(values) else np.nan
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _stage4_persistence_review(
    cells: pd.DataFrame,
    stage1: Stage1Config,
    config: Stage4Config,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for support_threshold in (
        stage1.primary_visualization_min_moving_count,
        stage1.sensitivity_visualization_min_moving_count,
    ):
        support = cells.N_out_move.ge(support_threshold)
        persistence = cells[f"C_neigh_out_1_ge_{support_threshold}"]
        for candidate_field in (
            "U_out_all_magnitude_rate",
            "U_coh_rate",
        ):
            for quantile in config.flux_percentiles:
                label = round(100 * quantile)
                selected = cells[
                    f"stage4_{'U_out_all' if candidate_field.startswith('U_out_all') else 'U_coh'}_ge_q{label}_support_{support_threshold}"
                ]
                records.append(
                    {
                        "support_threshold": support_threshold,
                        "candidate_field": candidate_field,
                        "quantile": quantile,
                        "n_selected": int(selected.sum()),
                        "n_persistence_defined": int(
                            (selected & persistence.notna()).sum()
                        ),
                        "median_C_neigh_out_1": float(
                            persistence.loc[selected].median()
                        ),
                        "fraction_C_neigh_out_1_ge_0_8": float(
                            persistence.loc[selected].ge(0.8).mean()
                        ),
                        "median_R1_out": float(cells.loc[selected, "R1_out"].median()),
                        "median_N_neigh_out_1": float(
                            cells.loc[
                                selected,
                                f"N_neigh_out_1_ge_{support_threshold}",
                            ].median()
                        ),
                    }
                )
        records.append(
            {
                "support_threshold": support_threshold,
                "candidate_field": "all_supported_cells",
                "quantile": np.nan,
                "n_selected": int(support.sum()),
                "n_persistence_defined": int((support & persistence.notna()).sum()),
                "median_C_neigh_out_1": float(persistence.loc[support].median()),
                "fraction_C_neigh_out_1_ge_0_8": float(
                    persistence.loc[support].ge(0.8).mean()
                ),
                "median_R1_out": float(cells.loc[support, "R1_out"].median()),
                "median_N_neigh_out_1": float(
                    cells.loc[support, f"N_neigh_out_1_ge_{support_threshold}"].median()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _stage4_low_alignment_review(
    cells: pd.DataFrame,
    stage1: Stage1Config,
    stage2: Stage2Config,
    stage3b: Stage3BConfig,
    config: Stage4Config,
) -> pd.DataFrame:
    primary = stage1.primary_visualization_min_moving_count
    joint_primary = cells.N_out_move.ge(primary) & cells.N_in_move.ge(primary)
    local_reliable = cells.R1_out.ge(stage2.high_R1) & cells.R1_in.ge(stage2.high_R1)
    spatially_persistent = cells[f"C_neigh_out_1_ge_{primary}"].ge(
        stage3b.diagnostic_high_neighborhood_consistency
    ) & cells[f"C_neigh_in_1_ge_{primary}"].ge(
        stage3b.diagnostic_high_neighborhood_consistency
    )
    outward_complex = (
        cells.R1_out.between(stage2.split_R1_min, stage2.split_R1_max)
        & cells.R2_out.le(stage2.split_R2_max)
    ) | (cells.R1_out.le(stage2.low_R1) & cells.R2_out.ge(stage2.high_R2))
    incoming_complex = (
        cells.R1_in.between(stage2.split_R1_min, stage2.split_R1_max)
        & cells.R2_in.le(stage2.split_R2_max)
    ) | (cells.R1_in.le(stage2.low_R1) & cells.R2_in.ge(stage2.high_R2))
    low_alignment = cells.abs_delta_theta_io_1.ge(
        config.low_alignment_turn_degrees
    ) | cells.abs_delta_theta_io_mu.ge(config.low_alignment_turn_degrees)
    review = cells.loc[low_alignment].copy()
    review["stage4_joint_primary_support"] = joint_primary.loc[review.index]
    review["stage4_local_in_out_reliable"] = local_reliable.loc[review.index]
    review["stage4_in_out_spatially_persistent"] = spatially_persistent.loc[
        review.index
    ]
    review["stage4_outward_complex_geometry_evidence"] = outward_complex.loc[
        review.index
    ]
    review["stage4_incoming_complex_geometry_evidence"] = incoming_complex.loc[
        review.index
    ]
    interpretations: list[str] = []
    reasons: list[str] = []
    for row in review.itertuples():
        abs_turn = max(
            value
            for value in (row.abs_delta_theta_io_1, row.abs_delta_theta_io_mu)
            if np.isfinite(value)
        )
        if not row.stage4_joint_primary_support:
            interpretation = "insufficient_primary_joint_support"
            reason = (
                "N_out_move or N_in_move is below the primary visualization support"
            )
        elif (
            row.stage4_local_in_out_reliable and row.stage4_in_out_spatially_persistent
        ):
            if abs_turn >= config.reversal_like_turn_degrees:
                interpretation = "coherent_reversal_like_turning_geometry"
                reason = "large near-reversal turn with reliable local directions and persistent neighborhoods"
            else:
                interpretation = "coherent_turning_geometry"
                reason = "substantial turn with reliable local directions and persistent neighborhoods"
        elif (
            row.stage4_outward_complex_geometry_evidence
            or row.stage4_incoming_complex_geometry_evidence
        ):
            interpretation = "possible_split_merge_or_axial_geometry"
            reason = "R1/R2 combination indicates complex angular geometry; no topology inferred"
        elif row.R1_out < stage2.high_R1 or row.R1_in < stage2.high_R1:
            interpretation = "directionally_ambiguous_turning"
            reason = "one or both local mean directions have sub-high R1"
        else:
            interpretation = "supported_unresolved_turning_geometry"
            reason = "supported turning does not satisfy the stricter reliability descriptions"
        interpretations.append(interpretation)
        reasons.append(reason)
    review.insert(0, "stage4_low_alignment_interpretation", interpretations)
    review.insert(1, "stage4_interpretation_reason", reasons)
    review.insert(2, "stage4_interpretation_is_provisional", True)
    retained = [
        "stage4_low_alignment_interpretation",
        "stage4_interpretation_reason",
        "stage4_interpretation_is_provisional",
        "cell_id",
        "x",
        "y",
        "N_out_move",
        "N_in_move",
        "U_out_all_magnitude_rate",
        "U_coh_rate",
        "R1_out",
        "R2_out",
        "delta_theta_mu1_out",
        "R1_in",
        "R2_in",
        "delta_theta_mu1_in",
        f"C_neigh_out_1_ge_{primary}",
        f"N_neigh_out_1_ge_{primary}",
        f"C_neigh_in_1_ge_{primary}",
        f"N_neigh_in_1_ge_{primary}",
        "delta_theta_io_1",
        "delta_theta_io_mu",
        "A_io",
        "A_io_mu",
        "stage4_joint_primary_support",
        "stage4_local_in_out_reliable",
        "stage4_in_out_spatially_persistent",
        "stage4_outward_complex_geometry_evidence",
        "stage4_incoming_complex_geometry_evidence",
    ]
    return review.loc[:, retained].sort_values(
        ["stage4_low_alignment_interpretation", "U_out_all_magnitude_rate"],
        ascending=[True, False],
        kind="stable",
    )


def _stage4_representatives(
    cells: pd.DataFrame,
    low_alignment_review: pd.DataFrame,
    stage1: Stage1Config,
    stage2: Stage2Config,
    config: Stage4Config,
) -> pd.DataFrame:
    primary = stage1.primary_visualization_min_moving_count
    q95_u = cells[f"stage4_U_out_all_ge_q95_support_{primary}"]
    q95_coh = cells[f"stage4_U_coh_ge_q95_support_{primary}"]
    specifications: list[tuple[str, pd.Series, pd.Series, str, pd.Series]] = [
        (
            "flux_backbone_stable_under_coherence_weighting",
            q95_u & q95_coh,
            cells.U_out_all_magnitude_rate,
            "top-5% flux under both raw and coherence-weighted comparisons",
            q95_u,
        ),
        (
            "coherence_weighting_suppresses_flux_cell",
            q95_u & ~q95_coh,
            cells.U_out_all_magnitude_rate * (1.0 - cells.R1_out),
            "top-5% raw flux omitted from top-5% U_coh",
            q95_u,
        ),
        (
            "strong_direction_convention_disagreement",
            cells.diagnostic_strong_U_out_all
            & cells.R1_out.ge(stage2.high_R1)
            & cells.N_out_move.ge(primary)
            & cells.delta_theta_mu1_out.ge(config.direction_difference_review_degrees),
            cells.U_out_all_magnitude_rate * cells.delta_theta_mu1_out,
            "strong reliable outgoing flux with material theta_mu/theta1 disagreement",
            cells.diagnostic_strong_U_out_all
            & cells.R1_out.ge(stage2.high_R1)
            & cells.N_out_move.ge(primary),
        ),
    ]
    outputs: list[pd.DataFrame] = []
    for category, mask, score, rule, fallback_pool in specifications:
        candidates = cells.loc[mask].copy()
        fallback = False
        if candidates.empty:
            candidates = cells.loc[fallback_pool].copy()
            fallback = True
        candidates["selection_score"] = score.loc[candidates.index]
        candidates = candidates.sort_values(
            ["selection_score", "U_out_all_magnitude_rate"],
            ascending=False,
            kind="stable",
        ).head(config.representative_cells_per_category)
        candidates.insert(0, "stage4_representative_category", category)
        candidates.insert(
            1, "stage4_representative_rank", np.arange(1, len(candidates) + 1)
        )
        candidates.insert(2, "selection_rule", rule)
        candidates.insert(3, "selection_used_fallback", fallback)
        outputs.append(candidates)
    for interpretation, group in low_alignment_review.groupby(
        "stage4_low_alignment_interpretation", sort=True
    ):
        selected_ids = group.nlargest(
            config.representative_cells_per_category,
            "U_out_all_magnitude_rate",
        ).cell_id
        candidates = (
            cells.loc[cells.cell_id.isin(selected_ids)]
            .copy()
            .sort_values("U_out_all_magnitude_rate", ascending=False, kind="stable")
        )
        candidates["selection_score"] = candidates.U_out_all_magnitude_rate
        candidates.insert(
            0, "stage4_representative_category", f"low_alignment_{interpretation}"
        )
        candidates.insert(
            1, "stage4_representative_rank", np.arange(1, len(candidates) + 1)
        )
        candidates.insert(
            2,
            "selection_rule",
            "highest flux within provisional low-alignment interpretation",
        )
        candidates.insert(3, "selection_used_fallback", False)
        outputs.append(candidates)
    representatives = pd.concat(outputs, ignore_index=True)
    retained = [
        "stage4_representative_category",
        "stage4_representative_rank",
        "selection_rule",
        "selection_used_fallback",
        "selection_score",
        "cell_id",
        "x",
        "y",
        "N_out_move",
        "N_in_move",
        "U_out_all_magnitude_rate",
        "U_coh_rate",
        "R1_out",
        "R2_out",
        "theta_mu_out",
        "theta1_out",
        "delta_theta_mu1_out",
        "R1_in",
        "R2_in",
        "theta_mu_in_motion_destination",
        "theta1_in_motion_destination",
        "delta_theta_mu1_in",
        f"C_neigh_out_1_ge_{primary}",
        f"C_neigh_out_mu_ge_{primary}",
        f"C_neigh_in_1_ge_{primary}",
        f"C_neigh_in_mu_ge_{primary}",
        "delta_theta_io_1",
        "delta_theta_io_mu",
        "A_io",
        "A_io_mu",
    ]
    return representatives.loc[:, retained]


def compute_stage4_fields(
    stage3b_cells: pd.DataFrame,
    grid: GridConfig,
    *,
    stage1: Stage1Config,
    stage2: Stage2Config,
    stage3b: Stage3BConfig,
    config: Stage4Config,
) -> Stage4Fields:
    """Review interpretable candidate-backbone and independent diagnostic fields."""
    required = {
        "cell_id",
        "N_out_move",
        "N_in_move",
        "U_out_all_magnitude_rate",
        "theta_mu_out",
        "theta1_out",
        "R1_out",
        "R2_out",
        "delta_theta_mu1_out",
        "theta_mu_in_motion_destination",
        "theta1_in_motion_destination",
        "R1_in",
        "R2_in",
        "delta_theta_mu1_in",
        "A_io",
        "A_io_mu",
        "diagnostic_strong_U_out_all",
    }
    for threshold in (
        stage1.primary_visualization_min_moving_count,
        stage1.sensitivity_visualization_min_moving_count,
    ):
        required.update(
            {
                f"C_neigh_out_1_ge_{threshold}",
                f"N_neigh_out_1_ge_{threshold}",
                f"C_neigh_out_mu_ge_{threshold}",
                f"C_neigh_in_1_ge_{threshold}",
                f"N_neigh_in_1_ge_{threshold}",
                f"C_neigh_in_mu_ge_{threshold}",
            }
        )
    missing = sorted(required - set(stage3b_cells.columns))
    if missing:
        raise ValueError(f"Stage 3B fields missing Stage 4 columns: {missing}")
    cells = stage3b_cells.copy()
    cells["S_flux_rate"] = cells.U_out_all_magnitude_rate
    cells["U_coh_rate"] = cells.U_out_all_magnitude_rate * cells.R1_out
    cells["U_coh_fraction_of_flux"] = cells.R1_out
    cells["delta_theta_io_1"] = np.nan
    turn1_defined = (
        cells.theta1_out.notna() & cells.theta1_in_motion_destination.notna()
    )
    cells.loc[turn1_defined, "delta_theta_io_1"] = _signed_circular_difference_degrees(
        cells.loc[turn1_defined, "theta1_out"],
        cells.loc[turn1_defined, "theta1_in_motion_destination"],
    )
    cells["delta_theta_io_mu"] = np.nan
    turnmu_defined = (
        cells.theta_mu_out.notna() & cells.theta_mu_in_motion_destination.notna()
    )
    cells.loc[turnmu_defined, "delta_theta_io_mu"] = (
        _signed_circular_difference_degrees(
            cells.loc[turnmu_defined, "theta_mu_out"],
            cells.loc[turnmu_defined, "theta_mu_in_motion_destination"],
        )
    )
    cells["abs_delta_theta_io_1"] = cells.delta_theta_io_1.abs()
    cells["abs_delta_theta_io_mu"] = cells.delta_theta_io_mu.abs()
    cells["delta_theta_io_1_minus_mu"] = np.nan
    both_turn_defined = cells.delta_theta_io_1.notna() & cells.delta_theta_io_mu.notna()
    cells.loc[both_turn_defined, "delta_theta_io_1_minus_mu"] = (
        _signed_circular_difference_degrees(
            cells.loc[both_turn_defined, "delta_theta_io_1"],
            cells.loc[both_turn_defined, "delta_theta_io_mu"],
        )
    )
    cells["abs_delta_theta_io_1_minus_mu"] = cells.delta_theta_io_1_minus_mu.abs()
    cells["stage4_substantial_turn_1"] = cells.abs_delta_theta_io_1.ge(
        config.substantial_turn_degrees
    )
    cells["stage4_substantial_turn_mu"] = cells.abs_delta_theta_io_mu.ge(
        config.substantial_turn_degrees
    )

    flux_levels = _stage4_flux_level_comparison(cells, stage1, config)
    direction_comparison = _stage4_direction_comparison(cells, stage1, stage2, config)
    persistence_review = _stage4_persistence_review(cells, stage1, config)
    low_alignment_review = _stage4_low_alignment_review(
        cells, stage1, stage2, stage3b, config
    )
    representatives = _stage4_representatives(
        cells, low_alignment_review, stage1, stage2, config
    )

    primary = stage1.primary_visualization_min_moving_count
    sensitivity = stage1.sensitivity_visualization_min_moving_count
    summary: dict[str, Any] = {
        "primary_candidate_backbone_field": "U_out_all_magnitude_rate",
        "optional_comparison_field": "U_coh_rate = U_out_all_magnitude_rate * R1_out",
        "master_score_created": False,
        "branch_threshold_selected": False,
        "branch_extraction_implemented": False,
        "stage5_implemented": False,
        "low_alignment_interpretations_are_provisional": True,
        "geographic_or_current_specific_filters_applied": False,
        "turn_angle_convention": "wrap(theta_out - theta_in_arrival) to [-180, 180)",
    }
    for threshold in (primary, sensitivity):
        mask = cells.N_out_move.ge(threshold)
        joint = mask & cells.N_in_move.ge(threshold)
        locally_reliable_through = (
            joint & cells.R1_out.ge(stage2.high_R1) & cells.R1_in.ge(stage2.high_R1)
        )
        spatially_persistent_through = cells[f"C_neigh_out_1_ge_{threshold}"].ge(
            stage3b.diagnostic_high_neighborhood_consistency
        ) & cells[f"C_neigh_in_1_ge_{threshold}"].ge(
            stage3b.diagnostic_high_neighborhood_consistency
        )
        reliable_persistent_through = (
            locally_reliable_through & spatially_persistent_through
        )
        reliable_out = mask & cells.R1_out.ge(stage2.high_R1)
        strong_reliable_out = reliable_out & cells.diagnostic_strong_U_out_all
        strong_reliable_through = (
            strong_reliable_out
            & cells.R1_in.ge(stage2.high_R1)
            & cells.N_in_move.ge(threshold)
        )
        rank_pair = cells.loc[mask, ["U_out_all_magnitude_rate", "U_coh_rate"]]
        summary.update(
            {
                f"ge_{threshold}_spearman_U_out_all_vs_U_coh": float(
                    rank_pair.corr(method="spearman").iloc[0, 1]
                ),
                f"ge_{threshold}_median_R1_out": float(
                    cells.loc[mask, "R1_out"].median()
                ),
                f"ge_{threshold}_strong_reliable_out_cells": int(
                    strong_reliable_out.sum()
                ),
                f"ge_{threshold}_strong_reliable_out_median_delta_theta_mu1_out": float(
                    cells.loc[strong_reliable_out, "delta_theta_mu1_out"].median()
                ),
                f"ge_{threshold}_strong_reliable_out_q95_delta_theta_mu1_out": float(
                    cells.loc[strong_reliable_out, "delta_theta_mu1_out"].quantile(0.95)
                ),
                f"ge_{threshold}_strong_reliable_out_cells_delta_theta_mu1_out_ge_review": int(
                    (
                        strong_reliable_out
                        & cells.delta_theta_mu1_out.ge(
                            config.direction_difference_review_degrees
                        )
                    ).sum()
                ),
                f"ge_{threshold}_strong_reliable_through_cells": int(
                    strong_reliable_through.sum()
                ),
                f"ge_{threshold}_strong_reliable_through_substantial_turn_1_cells": int(
                    (
                        strong_reliable_through
                        & cells.abs_delta_theta_io_1.ge(config.substantial_turn_degrees)
                    ).sum()
                ),
                f"ge_{threshold}_joint_cells": int(joint.sum()),
                f"ge_{threshold}_reliable_persistent_through_cells": int(
                    reliable_persistent_through.sum()
                ),
                f"ge_{threshold}_reliable_persistent_through_substantial_turn_1_cells": int(
                    (
                        reliable_persistent_through
                        & cells.abs_delta_theta_io_1.ge(config.substantial_turn_degrees)
                    ).sum()
                ),
                f"ge_{threshold}_reliable_persistent_through_low_alignment_1_cells": int(
                    (
                        reliable_persistent_through
                        & cells.abs_delta_theta_io_1.ge(
                            config.low_alignment_turn_degrees
                        )
                    ).sum()
                ),
                f"ge_{threshold}_reliable_persistent_through_reversal_like_1_cells": int(
                    (
                        reliable_persistent_through
                        & cells.abs_delta_theta_io_1.ge(
                            config.reversal_like_turn_degrees
                        )
                    ).sum()
                ),
                f"ge_{threshold}_low_alignment_review_cells": int(
                    (
                        joint
                        & (
                            cells.abs_delta_theta_io_1.ge(
                                config.low_alignment_turn_degrees
                            )
                            | cells.abs_delta_theta_io_mu.ge(
                                config.low_alignment_turn_degrees
                            )
                        )
                    ).sum()
                ),
                f"ge_{threshold}_A_io_cosine_identity_max_abs": float(
                    (
                        cells.loc[joint, "A_io"]
                        - np.cos(np.deg2rad(cells.loc[joint, "delta_theta_io_1"]))
                    )
                    .abs()
                    .max()
                ),
                f"ge_{threshold}_A_io_mu_cosine_identity_max_abs": float(
                    (
                        cells.loc[joint, "A_io_mu"]
                        - np.cos(np.deg2rad(cells.loc[joint, "delta_theta_io_mu"]))
                    )
                    .abs()
                    .max()
                ),
            }
        )
    interpretation_counts = (
        low_alignment_review.groupby("stage4_low_alignment_interpretation")
        .size()
        .to_dict()
    )
    summary["low_alignment_interpretation_counts_all_support"] = interpretation_counts
    for threshold in (primary, sensitivity):
        supported_review = low_alignment_review.loc[
            low_alignment_review.N_out_move.ge(threshold)
            & low_alignment_review.N_in_move.ge(threshold)
        ]
        summary[f"ge_{threshold}_low_alignment_interpretation_counts"] = (
            supported_review.groupby("stage4_low_alignment_interpretation")
            .size()
            .to_dict()
        )
    dataset = _stage1_dataset(cells, grid).assign_attrs(
        stage4_primary_backbone="U_out_all_magnitude_rate",
        stage4_optional_comparison="U_coh_rate",
        stage4_master_score="not created",
        stage4_branch_threshold="not selected",
        stage4_branch_extraction="not implemented",
        stage5="not implemented",
        stage4_turn_angle_convention="theta_out minus theta_in_arrival, wrapped [-180,180)",
        stage4_A_io_role="cosine representation of local turning, not branch quality",
        geographic_filters="none",
    )
    return Stage4Fields(
        cells=cells,
        dataset=dataset,
        flux_levels=flux_levels,
        direction_comparison=direction_comparison,
        persistence_review=persistence_review,
        low_alignment_review=low_alignment_review,
        representatives=representatives,
        summary=summary,
    )
