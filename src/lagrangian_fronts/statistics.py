"""Single-pass transition statistics used by the compact workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._statistics_kernel import (
    Stage1Config,
    Stage2Config,
    Stage3Config,
    compute_stage1_fields,
    compute_stage2_fields,
    compute_stage3_fields,
    compute_support_fields,
    validate_transition_table,
)
from .config import AnalysisConfig
from .geometry import make_spatial_geometry

GEOGRAPHIC_INPUT_COLUMNS = {
    "start_lon_bin": "start_x_bin",
    "start_lat_bin": "start_y_bin",
    "end_lon_bin": "end_x_bin",
    "end_lat_bin": "end_y_bin",
    "start_lon_center": "start_x_center",
    "start_lat_center": "start_y_center",
    "end_lon_center": "end_x_center",
    "end_lat_center": "end_y_center",
    "transition_count": "transition_count",
    "transition_probability": "transition_probability",
}
CARTESIAN_INPUT_COLUMNS = {
    "start_x_bin": "start_x_bin",
    "start_y_bin": "start_y_bin",
    "end_x_bin": "end_x_bin",
    "end_y_bin": "end_y_bin",
    "start_x_center": "start_x_center",
    "start_y_center": "start_y_center",
    "end_x_center": "end_x_center",
    "end_y_center": "end_y_center",
    "transition_count": "transition_count",
    "transition_probability": "transition_probability",
}


@dataclass(frozen=True)
class TransitionStatistics:
    links: pd.DataFrame
    cells: pd.DataFrame
    validation_summary: dict


def normalize_transition_table(
    table: pd.DataFrame, coordinate_system: str
) -> pd.DataFrame:
    """Validate the mode-specific schema and return canonical internal x/y fields."""
    columns = (
        GEOGRAPHIC_INPUT_COLUMNS
        if coordinate_system == "geographic"
        else CARTESIAN_INPUT_COLUMNS
    )
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(
            "Transition-matrix validation failed: "
            f"['missing_columns:{','.join(missing)}']"
        )
    return table.loc[:, list(columns)].rename(columns=columns).copy()


def _directional_vector_fields(
    cells: pd.DataFrame, *, zero_tolerance: float
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Expose the distance-free first harmonic as moving/all-population vectors."""
    required = {
        "M1_out_real",
        "M1_out_imag",
        "P_move",
        "R1_out",
        "theta1_out",
    }
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"Directional first-harmonic fields missing columns: {missing}")

    output = cells.copy()
    # Bearings are clockwise from positive y, so the harmonic's imaginary
    # component is x-directed and its real component is y-directed.
    output["D_out_move_x"] = output.M1_out_imag
    output["D_out_move_y"] = output.M1_out_real
    output["D_out_move_magnitude"] = np.hypot(
        output.D_out_move_x, output.D_out_move_y
    )
    output["D_out_all_x"] = output.P_move * output.D_out_move_x
    output["D_out_all_y"] = output.P_move * output.D_out_move_y
    output["D_out_all_magnitude"] = np.hypot(
        output.D_out_all_x, output.D_out_all_y
    )

    # A source population containing only stay transitions has a zero
    # full-population directional vector, while its moving-conditioned vector
    # and bearing remain undefined.
    no_movement = output.P_move.eq(0.0) & output.P_move.notna()
    output.loc[
        no_movement,
        ["D_out_all_x", "D_out_all_y", "D_out_all_magnitude"],
    ] = 0.0

    output["directional_identity_x_residual"] = (
        output.D_out_all_x - output.P_move * output.D_out_move_x
    )
    output["directional_identity_y_residual"] = (
        output.D_out_all_y - output.P_move * output.D_out_move_y
    )
    output["directional_move_magnitude_minus_R1"] = (
        output.D_out_move_magnitude - output.R1_out
    )
    output["directional_all_magnitude_minus_P_move_R1"] = (
        output.D_out_all_magnitude - output.P_move * output.R1_out
    )

    move_bearing = np.full(len(output), np.nan, dtype=float)
    move_defined = output.D_out_move_magnitude.gt(zero_tolerance)
    move_bearing[move_defined] = np.remainder(
        np.rad2deg(
            np.arctan2(
                output.loc[move_defined, "D_out_move_x"],
                output.loc[move_defined, "D_out_move_y"],
            )
        ),
        360.0,
    )
    all_bearing = np.full(len(output), np.nan, dtype=float)
    all_defined = output.D_out_all_magnitude.gt(zero_tolerance)
    all_bearing[all_defined] = np.remainder(
        np.rad2deg(
            np.arctan2(
                output.loc[all_defined, "D_out_all_x"],
                output.loc[all_defined, "D_out_all_y"],
            )
        ),
        360.0,
    )
    output["D_out_move_bearing_residual_degrees"] = np.nan
    output["D_out_all_bearing_residual_degrees"] = np.nan
    theta_defined = output.theta1_out.notna()
    move_compare = move_defined & theta_defined
    all_compare = all_defined & theta_defined
    output.loc[move_compare, "D_out_move_bearing_residual_degrees"] = np.remainder(
        move_bearing[move_compare] - output.loc[move_compare, "theta1_out"] + 180.0,
        360.0,
    ) - 180.0
    output.loc[all_compare, "D_out_all_bearing_residual_degrees"] = np.remainder(
        all_bearing[all_compare] - output.loc[all_compare, "theta1_out"] + 180.0,
        360.0,
    ) - 180.0

    def maximum_absolute(field: str) -> float:
        values = output[field].dropna().abs()
        return float(values.max()) if not values.empty else np.nan

    summary: dict[str, float | int] = {
        "cells_with_D_out_move": int(output.D_out_move_magnitude.notna().sum()),
        "cells_with_D_out_all": int(output.D_out_all_magnitude.notna().sum()),
        "cells_with_defined_directional_bearing": int(theta_defined.sum()),
        "D_out_all_equals_P_move_D_out_move_max_abs_x": maximum_absolute(
            "directional_identity_x_residual"
        ),
        "D_out_all_equals_P_move_D_out_move_max_abs_y": maximum_absolute(
            "directional_identity_y_residual"
        ),
        "D_out_move_magnitude_equals_R1_max_abs": maximum_absolute(
            "directional_move_magnitude_minus_R1"
        ),
        "D_out_all_magnitude_equals_P_move_R1_max_abs": maximum_absolute(
            "directional_all_magnitude_minus_P_move_R1"
        ),
        "D_out_move_bearing_equals_theta1_max_abs_degrees": maximum_absolute(
            "D_out_move_bearing_residual_degrees"
        ),
        "D_out_all_bearing_equals_theta1_max_abs_degrees": maximum_absolute(
            "D_out_all_bearing_residual_degrees"
        ),
    }
    return output, summary


def compute_transition_statistics(
    table: pd.DataFrame, config: AnalysisConfig
) -> TransitionStatistics:
    """Validate the normalized matrix and calculate each retained statistic once."""
    normalized = normalize_transition_table(table, config.geometry.coordinate_system)
    validation = validate_transition_table(
        normalized,
        config.grid,
        config.validation,
    )
    if validation.errors:
        raise ValueError(f"Transition-matrix validation failed: {validation.errors}")
    support = compute_support_fields(
        validation.links, config.grid, (config.statistics.min_moving_support,)
    )
    stage1_config = Stage1Config(
        primary_visualization_min_moving_count=config.statistics.min_moving_support,
        sensitivity_visualization_min_moving_count=config.statistics.min_moving_support,
        direction_zero_tolerance=config.statistics.direction_zero_tolerance,
    )
    outward = compute_stage1_fields(
        validation.links,
        support.cells,
        config.grid,
        timestep=config.matrix.timestep,
        geometry=make_spatial_geometry(config.geometry),
        config=stage1_config,
    )
    stage2_config = Stage2Config(
        angular_bins=config.statistics.angular_bins,
        harmonic_zero_tolerance=config.statistics.direction_zero_tolerance,
        high_R1=config.statistics.high_R1,
        low_R1=config.statistics.low_R1,
    )
    outgoing = compute_stage2_fields(
        outward.links,
        outward.cells,
        config.grid,
        stage1=stage1_config,
        config=stage2_config,
    )
    incoming = compute_stage3_fields(
        outward.links,
        outgoing.cells,
        config.grid,
        stage1=stage1_config,
        stage2=stage2_config,
        config=Stage3Config(
            angular_bins=config.statistics.angular_bins,
            harmonic_zero_tolerance=config.statistics.direction_zero_tolerance,
        ),
    )
    cells, links = incoming.cells, incoming.links
    cells, directional_summary = _directional_vector_fields(
        cells,
        zero_tolerance=config.statistics.direction_zero_tolerance,
    )
    cells["S_flux_rate"] = cells.U_out_all_magnitude_rate
    if {"theta1_out", "theta1_in_motion_destination"} <= set(cells):
        cells["delta_theta_io"] = (
            np.remainder(
                cells.theta1_out - cells.theta1_in_motion_destination + 180.0, 360.0
            )
            - 180.0
        )
    else:
        cells["delta_theta_io"] = np.nan
    cells["delta_theta_io_1"] = cells.delta_theta_io
    if "theta_mu_in_motion_destination" in cells:
        cells["delta_theta_io_mu"] = (
            np.remainder(
                cells.theta_mu_out - cells.theta_mu_in_motion_destination + 180.0,
                360.0,
            )
            - 180.0
        )
    else:
        cells["delta_theta_io_mu"] = np.nan
    validation_summary = dict(validation.summary)
    validation_summary["directional_vector_identities"] = directional_summary
    return TransitionStatistics(
        links=links, cells=cells, validation_summary=validation_summary
    )


COMPACT_CELL_FIELDS = (
    "cell_id",
    "x_bin",
    "y_bin",
    "x",
    "y",
    "N_out_move",
    "N_in_move",
    "P_stay",
    "P_move",
    "U_out_all_x_rate",
    "U_out_all_y_rate",
    "U_out_all_magnitude_rate",
    "theta_mu_out",
    "theta1_out",
    "R1_out",
    "R2_out",
    "angular_entropy_out",
    "D_out_move_x",
    "D_out_move_y",
    "D_out_move_magnitude",
    "D_out_all_x",
    "D_out_all_y",
    "D_out_all_magnitude",
    "R1_in",
    "R2_in",
    "theta1_in_motion_destination",
    "theta_mu_in_motion_destination",
    "delta_theta_mu1_in",
    "delta_theta_io",
)


def compact_cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical scientist-facing cell table."""
    selected = [name for name in COMPACT_CELL_FIELDS if name in cells]
    return cells[selected].rename(
        columns={"x_bin": "start_x_bin", "y_bin": "start_y_bin"}
    )
