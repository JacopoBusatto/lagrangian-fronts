from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagrangian_fronts._statistics_kernel import (
    GeometryConfig,
    GridConfig,
    Stage1Config,
    Stage2Config,
    Stage3BConfig,
    Stage3Config,
    Stage4Config,
    ValidationConfig,
    _neighbor_directional_consistency,
    _signed_circular_difference_degrees,
    compute_stage1_fields,
    compute_stage2_fields,
    compute_stage3_fields,
    compute_stage4_fields,
    compute_support_fields,
    validate_transition_table,
)


def _grid() -> GridConfig:
    return GridConfig(
        lon_min=0.0,
        lon_max=3.0,
        lat_min=0.0,
        lat_max=2.0,
        dlon=1.0,
        dlat=1.0,
        periodic_longitude=False,
    )


def _valid_table() -> pd.DataFrame:
    rows = [
        # Cell (0, 0): 10 stay, 30 east.
        (0, 0, 0, 0, 10, 0.25),
        (0, 0, 1, 0, 30, 0.75),
        # Cell (1, 0): 20 east, 20 north.
        (1, 0, 2, 0, 20, 0.50),
        (1, 0, 1, 1, 20, 0.50),
    ]
    return pd.DataFrame(
        {
            "start_x_bin": pd.Series([row[0] for row in rows], dtype="int64"),
            "start_y_bin": pd.Series([row[1] for row in rows], dtype="int64"),
            "end_x_bin": pd.Series([row[2] for row in rows], dtype="int64"),
            "end_y_bin": pd.Series([row[3] for row in rows], dtype="int64"),
            "start_x_center": pd.Series(
                [row[0] + 0.5 for row in rows], dtype="float64"
            ),
            "start_y_center": pd.Series(
                [row[1] + 0.5 for row in rows], dtype="float64"
            ),
            "end_x_center": pd.Series(
                [row[2] + 0.5 for row in rows], dtype="float64"
            ),
            "end_y_center": pd.Series(
                [row[3] + 0.5 for row in rows], dtype="float64"
            ),
            "transition_count": pd.Series([row[4] for row in rows], dtype="int64"),
            "transition_probability": pd.Series(
                [row[5] for row in rows], dtype="float64"
            ),
        }
    )


def test_stage0_validation_accepts_exact_count_normalization() -> None:
    result = validate_transition_table(_valid_table(), _grid(), ValidationConfig())

    assert result.errors == ()
    assert result.summary["n_sparse_links"] == 4
    assert result.summary["total_transition_count"] == 80
    assert result.summary["populated_start_cells"] == 2
    assert result.summary["normalization_residual_max_abs"] == 0.0
    assert result.invalid_links.empty


def test_stage0_validation_never_renormalizes_invalid_probabilities() -> None:
    table = _valid_table()
    table.loc[0, "transition_probability"] = 0.20
    original = table.transition_probability.copy()

    result = validate_transition_table(table, _grid(), ValidationConfig())

    assert "row_normalization_failure" in result.errors
    assert "count_probability_identity_failure" in result.errors
    pd.testing.assert_series_equal(result.links.transition_probability, original)
    assert np.isclose(
        result.rows.loc[result.rows.start_cell_id == 0, "sum_probability"].iloc[0],
        0.95,
    )


def test_stage0_validation_reports_duplicate_and_center_mismatch() -> None:
    table = pd.concat([_valid_table(), _valid_table().iloc[[0]]], ignore_index=True)
    table.loc[0, "start_x_center"] = 0.6

    result = validate_transition_table(table, _grid(), ValidationConfig())

    assert "duplicate_transition_keys" in result.errors
    assert "grid_center_mismatch" in result.errors
    assert result.summary["duplicate_link_rows"] == 2
    assert result.summary["center_mismatch_rows"] == 1


def test_stage0_validation_reports_missing_columns() -> None:
    table = _valid_table().drop(columns="transition_probability")

    result = validate_transition_table(table, _grid(), ValidationConfig())

    assert result.errors == ("missing_columns:transition_probability",)
    assert result.rows.empty


def test_support_fields_separate_total_moving_inward_and_outward_counts() -> None:
    validation = validate_transition_table(_valid_table(), _grid(), ValidationConfig())
    support = compute_support_fields(validation.links, _grid(), (10, 30, 50))
    cells = support.cells.set_index("cell_id")

    assert cells.loc[0, "N_out_total"] == 40
    assert cells.loc[0, "N_out_move"] == 30
    assert cells.loc[0, "C_stay"] == 10
    assert cells.loc[0, "P_stay"] == pytest.approx(0.25)
    assert cells.loc[0, "P_move"] == pytest.approx(0.75)
    assert cells.loc[1, "N_in_total"] == 30
    assert cells.loc[1, "N_in_move"] == 30
    assert cells.loc[1, "n_distinct_moving_sources"] == 1
    assert cells.loc[0, "n_distinct_moving_destinations"] == 1
    assert bool(cells.loc[0, "support_N_out_total_ge_30"])
    assert not bool(cells.loc[0, "support_N_out_total_ge_50"])


def test_destination_only_cell_has_undefined_stay_fraction() -> None:
    validation = validate_transition_table(_valid_table(), _grid(), ValidationConfig())
    support = compute_support_fields(validation.links, _grid(), (10,))
    cells = support.cells.set_index("cell_id")

    destination_only_cell = 2
    assert cells.loc[destination_only_cell, "N_out_total"] == 0
    assert cells.loc[destination_only_cell, "N_in_total"] == 20
    assert np.isnan(cells.loc[destination_only_cell, "P_stay"])
    assert np.isnan(cells.loc[destination_only_cell, "P_move"])


def test_support_coverage_reports_positive_union_and_domain_denominators() -> None:
    validation = validate_transition_table(_valid_table(), _grid(), ValidationConfig())
    support = compute_support_fields(validation.links, _grid(), (30,))
    row = support.coverage.loc[support.coverage.support_field.eq("N_out_total")].iloc[0]

    assert row.n_cells_above == 2
    assert row.n_positive_cells == 2
    assert row.n_union_cells == 4
    assert row.n_domain_cells == 6
    assert row.fraction_of_positive_cells == 1.0
    assert row.fraction_of_union_cells == 0.5
    assert row.fraction_of_domain_cells == pytest.approx(1 / 3)


def _stage1(table: pd.DataFrame, grid: GridConfig | None = None):
    selected_grid = _grid() if grid is None else grid
    validation = validate_transition_table(table, selected_grid, ValidationConfig())
    assert validation.errors == ()
    support = compute_support_fields(validation.links, selected_grid, (10, 20))
    return compute_stage1_fields(
        validation.links,
        support.cells,
        selected_grid,
        timestep=10.0,
        geometry=GeometryConfig(),
        config=Stage1Config(
            primary_visualization_min_moving_count=10,
            sensitivity_visualization_min_moving_count=20,
        ),
    )


def test_stage1_moving_and_total_moments_obey_stay_identity() -> None:
    result = _stage1(_valid_table())
    cell = result.cells.set_index("cell_id").loc[0]

    assert cell.mu_out_move_x_length > 100.0
    assert abs(cell.mu_out_move_y_length) < 0.1
    assert cell.mu_out_all_x_length == pytest.approx(
        cell.P_move * cell.mu_out_move_x_length, abs=1.0e-12
    )
    assert cell.U_out_all_magnitude_rate == pytest.approx(
        cell.P_move * cell.U_out_move_magnitude_rate, abs=1.0e-12
    )
    assert cell.U_out_retained_fraction == pytest.approx(cell.P_move)
    assert cell.theta_mu_out == pytest.approx(90.0, abs=0.1)
    assert cell.mean_moving_distance_length == pytest.approx(
        cell.mu_out_move_magnitude_length, rel=1.0e-6
    )
    assert (
        result.cells.theta_mu_out.dropna().between(0.0, 360.0, inclusive="left").all()
    )
    assert (
        result.links.source_forward_bearing.dropna()
        .between(0.0, 360.0, inclusive="left")
        .all()
    )


def test_stage1_fields_are_unmasked_below_visualization_support() -> None:
    table = _valid_table().iloc[[1]].copy()
    table["transition_count"] = 1
    table["transition_probability"] = 1.0
    result = _stage1(table)
    cell = result.cells.set_index("cell_id").loc[0]

    assert cell.N_out_move == 1
    assert not bool(cell.support_N_out_move_ge_10)
    assert np.isfinite(cell.U_out_move_magnitude_rate)
    assert np.isfinite(cell.U_out_all_magnitude_rate)
    assert "support_N_out_move_ge_10" in result.cells


def test_stage1_all_stay_cell_has_zero_total_and_undefined_moving_moment() -> None:
    table = _valid_table().iloc[[0]].copy()
    table["transition_probability"] = 1.0
    result = _stage1(table)
    cell = result.cells.iloc[0]

    assert cell.N_out_move == 0
    assert cell.mu_out_all_magnitude_length == 0.0
    assert cell.U_out_all_magnitude_rate == 0.0
    assert np.isnan(cell.mu_out_move_magnitude_length)
    assert np.isnan(cell.theta_mu_out)


def test_stage1_opposing_equal_moves_have_undefined_resultant_direction() -> None:
    table = pd.DataFrame(
        {
            "start_x_bin": pd.Series([1, 1], dtype="int64"),
            "start_y_bin": pd.Series([0, 0], dtype="int64"),
            "end_x_bin": pd.Series([0, 2], dtype="int64"),
            "end_y_bin": pd.Series([0, 0], dtype="int64"),
            "start_x_center": pd.Series([1.5, 1.5], dtype="float64"),
            "start_y_center": pd.Series([0.5, 0.5], dtype="float64"),
            "end_x_center": pd.Series([0.5, 2.5], dtype="float64"),
            "end_y_center": pd.Series([0.5, 0.5], dtype="float64"),
            "transition_count": pd.Series([10, 10], dtype="int64"),
            "transition_probability": pd.Series([0.5, 0.5], dtype="float64"),
        }
    )
    grid = GridConfig(
        lon_min=0.0,
        lon_max=3.0,
        lat_min=-0.5,
        lat_max=0.5,
        dlon=1.0,
        dlat=1.0,
        periodic_longitude=False,
    )
    table["start_y_center"] = 0.0
    table["end_y_center"] = 0.0
    validation = validate_transition_table(table, grid, ValidationConfig())
    support = compute_support_fields(validation.links, grid, (10,))
    result = compute_stage1_fields(
        validation.links,
        support.cells,
        grid,
        timestep=10.0,
        geometry=GeometryConfig(),
        config=Stage1Config(direction_zero_tolerance=1.0e-6),
    )
    cell = result.cells.loc[result.cells.cell_id == 1].iloc[0]

    assert cell.mu_out_move_magnitude_length < 1.0e-6
    assert np.isnan(cell.theta_mu_out)
    assert cell.mean_moving_distance_length > 100.0


def test_stage1_weighted_distance_quantiles_use_raw_counts() -> None:
    result = _stage1(_valid_table())
    cell = result.cells.set_index("cell_id").loc[1]
    links = result.links.loc[~result.links.is_stay & result.links.start_cell_id.eq(1)]

    assert cell.moving_distance_q25_length == pytest.approx(links.distance_length.min())
    assert cell.moving_distance_q75_length == pytest.approx(links.distance_length.max())


def test_stage1_dateline_geometry_uses_short_geodesic() -> None:
    grid = GridConfig(
        lon_min=-180.0,
        lon_max=180.0,
        lat_min=-1.0,
        lat_max=1.0,
        dlon=1.0,
        dlat=1.0,
        periodic_longitude=True,
    )
    table = pd.DataFrame(
        {
            "start_x_bin": pd.Series([359], dtype="int64"),
            "start_y_bin": pd.Series([1], dtype="int64"),
            "end_x_bin": pd.Series([0], dtype="int64"),
            "end_y_bin": pd.Series([1], dtype="int64"),
            "start_x_center": pd.Series([179.5], dtype="float64"),
            "start_y_center": pd.Series([0.5], dtype="float64"),
            "end_x_center": pd.Series([-179.5], dtype="float64"),
            "end_y_center": pd.Series([0.5], dtype="float64"),
            "transition_count": pd.Series([12], dtype="int64"),
            "transition_probability": pd.Series([1.0], dtype="float64"),
        }
    )
    result = _stage1(table, grid)
    link = result.links.iloc[0]

    assert 100.0 < link.distance_length < 120.0
    assert link.dx_source_length > 100.0
    assert link.source_forward_bearing == pytest.approx(90.0, abs=0.1)


def _prescribed_stage2(
    bearings: list[float],
    counts: list[int],
    distances: list[float],
):
    total = sum(counts)
    probabilities = np.asarray(counts, dtype=float) / total
    angle = np.deg2rad(np.asarray(bearings, dtype=float))
    east = float(np.sum(probabilities * np.asarray(distances) * np.sin(angle)))
    north = float(np.sum(probabilities * np.asarray(distances) * np.cos(angle)))
    theta_mu = float(np.remainder(np.rad2deg(np.arctan2(east, north)), 360.0))
    links = pd.DataFrame(
        {
            "start_cell_id": 0,
            "is_stay": False,
            "transition_count": pd.Series(counts, dtype="int64"),
            "conditional_moving_probability": probabilities,
            "source_forward_bearing": np.asarray(bearings, dtype=float),
            "distance_length": np.asarray(distances, dtype=float),
        }
    )
    weighted_q95 = (
        float(
            np.repeat(np.asarray(distances, dtype=float), np.asarray(counts, dtype=int))
            .reshape(-1)
            .tolist()[int(np.ceil(0.95 * total)) - 1]
        )
        if len(set(distances)) == 1
        else float(max(distances))
    )
    cells = pd.DataFrame(
        {
            "cell_id": [0],
            "x_bin": [0],
            "y_bin": [0],
            "x": [0.5],
            "y": [0.5],
            "N_out_total": [total],
            "N_out_move": [total],
            "P_stay": [0.0],
            "P_move": [1.0],
            "U_out_all_magnitude_rate": [float(np.hypot(east, north) / 10.0)],
            "U_out_move_magnitude_rate": [float(np.hypot(east, north) / 10.0)],
            "theta_mu_out": [theta_mu],
            "mean_moving_distance_length": [float(np.sum(probabilities * distances))],
            "moving_distance_q95_length": [weighted_q95],
            "diagnostic_strong_U_out_all": [True],
            "diagnostic_strong_U_out_move": [True],
        }
    )
    return compute_stage2_fields(
        links,
        cells,
        GridConfig(
            lon_min=0.0,
            lon_max=1.0,
            lat_min=0.0,
            lat_max=1.0,
            dlon=1.0,
            dlat=1.0,
            periodic_longitude=False,
        ),
        stage1=Stage1Config(),
        config=Stage2Config(angular_bins=36),
    )


def test_stage2_coherent_direction_has_unit_harmonics_and_zero_entropy() -> None:
    result = _prescribed_stage2([90.0], [12], [100.0])
    cell = result.cells.iloc[0]

    assert cell.R1_out == pytest.approx(1.0)
    assert cell.theta1_out == pytest.approx(90.0)
    assert cell.R2_out == pytest.approx(1.0)
    assert cell.theta2_out == pytest.approx(90.0)
    assert cell.angular_entropy_out == pytest.approx(0.0)
    assert cell.delta_theta_mu1_out == pytest.approx(0.0)


def test_stage2_symmetric_split_has_expected_first_and_second_harmonics() -> None:
    result = _prescribed_stage2([45.0, 135.0], [10, 10], [100.0, 100.0])
    cell = result.cells.iloc[0]

    assert cell.R1_out == pytest.approx(np.sqrt(0.5), abs=1.0e-12)
    assert cell.theta1_out == pytest.approx(90.0)
    assert cell.R2_out < 1.0e-12
    assert np.isnan(cell.theta2_out)
    assert cell.delta_theta_mu1_out == pytest.approx(0.0)


def test_stage2_opposite_axial_flow_has_zero_R1_and_unit_R2() -> None:
    result = _prescribed_stage2([90.0, 270.0], [10, 10], [100.0, 100.0])
    cell = result.cells.iloc[0]

    assert cell.R1_out < 1.0e-12
    assert np.isnan(cell.theta1_out)
    assert cell.R2_out == pytest.approx(1.0)
    assert cell.theta2_out == pytest.approx(90.0)
    assert np.isnan(cell.delta_theta_mu1_out)


def test_stage2_retains_long_link_and_detects_distance_weighted_disagreement() -> None:
    result = _prescribed_stage2([0.0, 90.0], [8, 2], [100.0, 1000.0])
    cell = result.cells.iloc[0]

    assert cell.moving_distance_max_length == 1000.0
    assert cell.theta1_out == pytest.approx(14.036243, abs=1.0e-6)
    assert cell.theta_mu_out == pytest.approx(68.198591, abs=1.0e-6)
    assert cell.delta_theta_mu1_out == pytest.approx(54.162348, abs=1.0e-6)
    assert cell.longest_link_distance_leverage_fraction == pytest.approx(2000 / 2800)


def test_stage2_entropy_uses_configured_fixed_bin_count() -> None:
    result = _prescribed_stage2([0.0, 90.0, 180.0, 270.0], [5, 5, 5, 5], [100.0] * 4)
    cell = result.cells.iloc[0]

    assert cell.angular_entropy_out == pytest.approx(np.log(4) / np.log(36))
    assert cell.n_occupied_angular_bins == 4
    assert result.dataset.attrs["stage2_angular_bins"] == 36
    assert result.summary["optional_angular_peak_diagnostic_implemented"] is False


def _prescribed_stage3(
    arrival_bearings: list[float],
    counts: list[int],
    distances: list[float],
    *,
    theta1_out: float = 90.0,
    theta_mu_out: float = 90.0,
):
    stage2_result = _prescribed_stage2([90.0], [20], [100.0])
    stage2_cells = stage2_result.cells.copy()
    stage2_cells["N_in_move"] = sum(counts)
    stage2_cells["theta1_out"] = theta1_out
    stage2_cells["theta_mu_out"] = theta_mu_out
    arrival = np.asarray(arrival_bearings, dtype=float)
    links = pd.DataFrame(
        {
            "start_cell_id": np.arange(100, 100 + len(counts)),
            "end_cell_id": 0,
            "is_stay": False,
            "transition_count": pd.Series(counts, dtype="int64"),
            "distance_length": np.asarray(distances, dtype=float),
            "source_forward_bearing": arrival,
            "theta_in_source": np.remainder(arrival + 180.0, 360.0),
            "theta_in_motion_destination": arrival,
            "conditional_moving_probability": 1.0,
        }
    )
    return compute_stage3_fields(
        links,
        stage2_cells,
        GridConfig(
            lon_min=0.0,
            lon_max=1.0,
            lat_min=0.0,
            lat_max=1.0,
            dlon=1.0,
            dlat=1.0,
            periodic_longitude=False,
        ),
        stage1=Stage1Config(),
        stage2=Stage2Config(angular_bins=36),
        config=Stage3Config(angular_bins=36),
    )


def test_stage3_coherent_arrival_uses_destination_local_motion_for_alignment() -> None:
    result = _prescribed_stage3([90.0], [12], [100.0])
    cell = result.cells.iloc[0]

    assert cell.R1_in == pytest.approx(1.0)
    assert cell.theta1_in_source == pytest.approx(270.0)
    assert cell.theta1_in_motion_destination == pytest.approx(90.0)
    assert cell.R2_in == pytest.approx(1.0)
    assert cell.theta2_in == pytest.approx(90.0)
    assert cell.H_in == pytest.approx(0.0)
    assert cell.theta_mu_in_motion_destination == pytest.approx(90.0)
    assert cell.delta_theta_mu1_in == pytest.approx(0.0)
    assert cell.A_io == pytest.approx(1.0)
    assert cell.A_io_mu == pytest.approx(1.0)


def test_stage3_opposite_incoming_motion_is_axial_and_has_no_mean_alignment() -> None:
    result = _prescribed_stage3([90.0, 270.0], [10, 10], [100.0, 100.0])
    cell = result.cells.iloc[0]

    assert cell.R1_in < 1.0e-12
    assert np.isnan(cell.theta1_in_source)
    assert np.isnan(cell.theta1_in_motion_destination)
    assert cell.R2_in == pytest.approx(1.0)
    assert cell.theta2_in == pytest.approx(90.0)
    assert np.isnan(cell.A_io)


def test_stage3_incoming_distance_weighting_retains_long_link_disagreement() -> None:
    result = _prescribed_stage3(
        [0.0, 90.0],
        [8, 2],
        [100.0, 1000.0],
        theta1_out=14.036243,
        theta_mu_out=0.0,
    )
    cell = result.cells.iloc[0]

    assert cell.incoming_moving_distance_max_length == 1000.0
    assert cell.theta1_in_motion_destination == pytest.approx(14.036243, abs=1.0e-6)
    assert cell.theta_mu_in_motion_destination == pytest.approx(68.198591, abs=1.0e-6)
    assert cell.delta_theta_mu1_in == pytest.approx(54.162348, abs=1.0e-6)
    assert cell.A_io == pytest.approx(1.0, abs=1.0e-12)
    assert cell.A_io_mu == pytest.approx(np.cos(np.deg2rad(68.198591)), abs=1.0e-6)


def test_stage3_entropy_uses_same_fixed_bins_as_stage2() -> None:
    result = _prescribed_stage3([0.0, 90.0, 180.0, 270.0], [5, 5, 5, 5], [100.0] * 4)
    cell = result.cells.iloc[0]

    assert cell.H_in == pytest.approx(np.log(4) / np.log(36))
    assert cell.n_occupied_incoming_angular_bins == 4
    assert result.dataset.attrs["stage3_angular_bins"] == 36
    assert result.summary["neighborhood_persistence_implemented"] is False


def _neighbor_cells(
    grid: GridConfig,
    populated: list[tuple[int, int]],
    directions: list[float],
    supports: list[int] | None = None,
) -> pd.DataFrame:
    if supports is None:
        supports = [20] * len(populated)
    return pd.DataFrame(
        {
            "cell_id": [lat * grid.nx + lon for lon, lat in populated],
            "x_bin": [lon for lon, _ in populated],
            "y_bin": [lat for _, lat in populated],
            "theta": directions,
            "support": supports,
        }
    )


def test_stage3b_uniform_eight_neighbor_consistency_and_count() -> None:
    grid = GridConfig(
        lon_min=0,
        lon_max=3,
        lat_min=0,
        lat_max=3,
        dlon=1,
        dlat=1,
        periodic_longitude=False,
    )
    populated = [(lon, lat) for lat in range(3) for lon in range(3)]
    cells = _neighbor_cells(grid, populated, [90.0] * 9)

    consistency, count = _neighbor_directional_consistency(
        cells,
        grid,
        direction_field="theta",
        support_field="support",
        support_threshold=10,
    )

    center = cells.index[cells.cell_id.eq(4)][0]
    assert count[center] == 8
    assert consistency[center] == pytest.approx(1.0)
    assert count[cells.index[cells.cell_id.eq(0)][0]] == 3


def test_stage3b_support_condition_changes_neighbor_set_without_bridging() -> None:
    grid = GridConfig(
        lon_min=0,
        lon_max=3,
        lat_min=0,
        lat_max=1,
        dlon=1,
        dlat=1,
        periodic_longitude=False,
    )
    cells = _neighbor_cells(
        grid,
        [(0, 0), (1, 0), (2, 0)],
        [90.0, 90.0, 270.0],
        [20, 20, 5],
    )

    unfiltered, unfiltered_count = _neighbor_directional_consistency(
        cells,
        grid,
        direction_field="theta",
        support_field="support",
        support_threshold=None,
    )
    supported, supported_count = _neighbor_directional_consistency(
        cells,
        grid,
        direction_field="theta",
        support_field="support",
        support_threshold=10,
    )

    middle = cells.index[cells.cell_id.eq(1)][0]
    assert unfiltered_count[middle] == 2
    assert unfiltered[middle] == pytest.approx(0.0)
    assert supported_count[middle] == 1
    assert supported[middle] == pytest.approx(1.0)


def test_stage3b_periodic_longitude_connects_only_immediate_wrapped_neighbor() -> None:
    grid = GridConfig(
        lon_min=-180,
        lon_max=180,
        lat_min=0,
        lat_max=1,
        dlon=120,
        dlat=1,
        periodic_longitude=True,
    )
    cells = _neighbor_cells(grid, [(0, 0), (2, 0)], [45.0, 45.0])

    consistency, count = _neighbor_directional_consistency(
        cells,
        grid,
        direction_field="theta",
        support_field="support",
        support_threshold=10,
    )

    assert np.array_equal(count, [1, 1])
    assert consistency == pytest.approx([1.0, 1.0])


def test_stage3b_one_valid_neighbor_remains_defined_with_explicit_count() -> None:
    grid = GridConfig(
        lon_min=0,
        lon_max=2,
        lat_min=0,
        lat_max=1,
        dlon=1,
        dlat=1,
        periodic_longitude=False,
    )
    cells = _neighbor_cells(grid, [(0, 0), (1, 0)], [0.0, 60.0])

    consistency, count = _neighbor_directional_consistency(
        cells,
        grid,
        direction_field="theta",
        support_field="support",
        support_threshold=20,
    )

    assert np.array_equal(count, [1, 1])
    assert consistency == pytest.approx([0.5, 0.5])


def _stage4_cells() -> pd.DataFrame:
    theta_out = np.array([10.0, 100.0, 170.0, 120.0])
    theta_in = np.zeros(4)
    return pd.DataFrame(
        {
            "cell_id": np.arange(4),
            "x_bin": [0, 1, 0, 1],
            "y_bin": [0, 0, 1, 1],
            "x": [0.5, 1.5, 0.5, 1.5],
            "y": [0.5, 0.5, 1.5, 1.5],
            "N_out_move": [25, 25, 25, 25],
            "N_in_move": [25, 25, 25, 25],
            "U_out_all_magnitude_rate": [2.0, 4.0, 8.0, 6.0],
            "theta_mu_out": theta_out + np.array([1.0, -2.0, 2.0, 5.0]),
            "theta1_out": theta_out,
            "R1_out": [0.9, 0.9, 0.9, 0.4],
            "R2_out": [0.8, 0.8, 0.8, 0.9],
            "delta_theta_mu1_out": [1.0, 2.0, 2.0, 5.0],
            "theta_mu_in_motion_destination": theta_in
            + np.array([0.0, 1.0, -1.0, 0.0]),
            "theta1_in_motion_destination": theta_in,
            "R1_in": [0.9, 0.9, 0.9, 0.4],
            "R2_in": [0.8, 0.8, 0.8, 0.9],
            "delta_theta_mu1_in": [0.0, 1.0, 1.0, 0.0],
            "A_io": np.cos(np.deg2rad(theta_out - theta_in)),
            "A_io_mu": np.cos(
                np.deg2rad(
                    theta_out
                    + np.array([1.0, -2.0, 2.0, 5.0])
                    - theta_in
                    - np.array([0.0, 1.0, -1.0, 0.0])
                )
            ),
            "diagnostic_strong_U_out_all": [False, False, True, True],
            "C_neigh_out_1_ge_10": [0.9] * 4,
            "N_neigh_out_1_ge_10": [3] * 4,
            "C_neigh_out_mu_ge_10": [0.9] * 4,
            "C_neigh_in_1_ge_10": [0.9] * 4,
            "N_neigh_in_1_ge_10": [3] * 4,
            "C_neigh_in_mu_ge_10": [0.9] * 4,
            "C_neigh_out_1_ge_20": [0.9] * 4,
            "N_neigh_out_1_ge_20": [3] * 4,
            "C_neigh_out_mu_ge_20": [0.9] * 4,
            "C_neigh_in_1_ge_20": [0.9] * 4,
            "N_neigh_in_1_ge_20": [3] * 4,
            "C_neigh_in_mu_ge_20": [0.9] * 4,
        }
    )


def test_stage4_signed_turn_wraps_to_half_open_interval() -> None:
    result = _signed_circular_difference_degrees(
        np.array([10.0, 350.0, 0.0, 180.0]),
        np.array([350.0, 10.0, 180.0, 0.0]),
    )

    assert result == pytest.approx([20.0, -20.0, -180.0, -180.0])


def test_stage4_retains_interpretable_fields_without_master_score_or_extraction() -> (
    None
):
    result = compute_stage4_fields(
        _stage4_cells(),
        GridConfig(
            lon_min=0.0,
            lon_max=2.0,
            lat_min=0.0,
            lat_max=2.0,
            dlon=1.0,
            dlat=1.0,
            periodic_longitude=False,
        ),
        stage1=Stage1Config(),
        stage2=Stage2Config(),
        stage3b=Stage3BConfig(),
        config=Stage4Config(),
    )
    cells = result.cells.set_index("cell_id")

    assert cells.loc[2, "U_coh_rate"] == pytest.approx(7.2)
    assert cells.loc[1, "delta_theta_io_1"] == pytest.approx(100.0)
    assert cells.loc[2, "delta_theta_io_1"] == pytest.approx(170.0)
    assert cells.loc[1, "A_io"] == pytest.approx(
        np.cos(np.deg2rad(cells.loc[1, "delta_theta_io_1"]))
    )
    assert result.summary["master_score_created"] is False
    assert result.summary["branch_threshold_selected"] is False
    assert result.summary["branch_extraction_implemented"] is False
    assert result.summary["stage5_implemented"] is False
    assert "coherent_turning_geometry" in set(
        result.low_alignment_review.stage4_low_alignment_interpretation
    )
    assert "coherent_reversal_like_turning_geometry" in set(
        result.low_alignment_review.stage4_low_alignment_interpretation
    )
    assert result.dataset.attrs["stage4_A_io_role"].endswith("not branch quality")
