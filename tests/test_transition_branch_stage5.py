from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagrangian_fronts._ridge_kernel import (
    extract_ridge_components,
    support_aware_uniform_3x3,
    transverse_ridge_diagnostics,
)
from lagrangian_fronts.config import FluxConfig, GridConfig


def _cells(
    grid: GridConfig,
    intensity: np.ndarray,
    theta: np.ndarray | float = 90.0,
    *,
    support: np.ndarray | int = 30,
    r1: np.ndarray | float = 0.9,
) -> pd.DataFrame:
    y_bin, x_bin = np.indices(intensity.shape)
    theta_values = np.broadcast_to(theta, intensity.shape).astype(float)
    support_values = np.broadcast_to(support, intensity.shape).astype(int)
    r1_values = np.broadcast_to(r1, intensity.shape).astype(float)
    frame = pd.DataFrame(
        {
            "cell_id": (y_bin * grid.nx + x_bin).ravel(),
            "x_bin": x_bin.ravel(),
            "y_bin": y_bin.ravel(),
            "x": (grid.x_min + (x_bin + 0.5) * grid.dlon).ravel(),
            "y": (grid.y_min + (y_bin + 0.5) * grid.dlat).ravel(),
            "N_out_move": support_values.ravel(),
            "N_in_move": support_values.ravel(),
            "U_out_all_magnitude_rate": intensity.ravel(),
            "theta_mu_out": theta_values.ravel(),
            "R1_out": r1_values.ravel(),
            "R2_out": np.full(intensity.size, 0.8),
            "delta_theta_mu1_out": np.full(intensity.size, 2.0),
            "R1_in": np.full(intensity.size, 0.9),
            "delta_theta_io_1": np.zeros(intensity.size),
            "delta_theta_io_mu": np.zeros(intensity.size),
            "C_neigh_out_for_experiment": np.full(intensity.size, 0.9),
        }
    )
    return frame


def _regular_grid(nx: int = 7, ny: int = 7) -> GridConfig:
    return GridConfig(
        lon_min=0.0,
        lon_max=float(nx),
        lat_min=-3.5,
        lat_max=-3.5 + ny,
        dlon=1.0,
        dlat=1.0,
        periodic_longitude=False,
    )


def _ridge(cells: pd.DataFrame, grid: GridConfig, variant: str = "raw") -> pd.DataFrame:
    result, _ = transverse_ridge_diagnostics(
        cells,
        grid,
        support_threshold=10,
        field_variant=variant,
        config=FluxConfig(),
    )
    return result


def _extract(
    cells: pd.DataFrame,
    grid: GridConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = cells.copy()
    cells["S0_field_rate"] = cells.U_out_all_magnitude_rate
    cells["C_perp_rate"] = 1.0
    cells["C_perp_normalized"] = 0.1
    cells["orientation_reliable_diagnostic"] = cells.R1_out.ge(0.8)
    cells["orientation_ambiguous_diagnostic"] = cells.R1_out.lt(0.5)
    cells["ridge_candidate"] = cells.get("ridge_candidate", False)
    cells["ridge_candidate_q90"] = cells.ridge_candidate
    return extract_ridge_components(
        cells,
        grid,
        support_threshold=10,
        field_variant="raw",
        intensity_level="q90",
        config=FluxConfig(),
    )


def test_stage5_straight_high_intensity_jet_is_a_transverse_ridge() -> None:
    grid = _regular_grid()
    intensity = np.ones((grid.ny, grid.nx))
    intensity[3, :] = 10.0
    result = _ridge(_cells(grid, intensity), grid)

    core = result.loc[result.y_bin.eq(3) & result.x_bin.between(1, 5)]
    assert core.ridge_candidate.all()
    assert core.C_perp_rate.min() > 8.0


def test_stage5_smoothly_curved_jet_retains_curved_core_cells() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    theta = np.full((7, 7), 90.0)
    path = [(1, 2), (2, 2), (3, 2), (4, 3), (5, 4)]
    for x_bin, y_bin in path:
        intensity[y_bin, x_bin] = 10.0
    theta[2, 3] = 45.0
    theta[3, 4] = 45.0
    theta[4, 5] = 30.0
    result = _ridge(_cells(grid, intensity, theta), grid).set_index("cell_id")

    assert all(result.loc[lat * 7 + lon, "ridge_candidate"] for lon, lat in path)


def test_stage5_relatively_sharp_coherent_bend_is_not_rejected() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    theta = np.full((7, 7), 90.0)
    path = [(1, 3), (2, 3), (3, 3), (3, 4), (3, 5)]
    for x_bin, y_bin in path:
        intensity[y_bin, x_bin] = 10.0
    theta[3, 3] = 45.0
    theta[4:, 3] = 0.0
    result = _ridge(_cells(grid, intensity, theta), grid).set_index("cell_id")

    assert result.loc[3 * 7 + 3, "ridge_candidate"]
    assert result.loc[5 * 7 + 3, "ridge_candidate"]


def test_stage5_two_parallel_jets_remain_two_components() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[2, :] = 10.0
    intensity[4, :] = 9.0
    diagnostics = _ridge(_cells(grid, intensity), grid)
    diagnostics["ridge_candidate_q90"] = (
        diagnostics.ridge_candidate & diagnostics.U_out_all_magnitude_rate.gt(5)
    )
    _, components, _, _ = extract_ridge_components(
        diagnostics,
        grid,
        support_threshold=10,
        field_variant="raw",
        intensity_level="q90",
        config=FluxConfig(),
    )

    assert len(components) == 2
    assert sorted(components.n_cells) == [7, 7]


def test_stage5_split_preserves_junction_without_selecting_continuation() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    cells = _cells(grid, intensity)
    selected = {(3, 1), (3, 2), (3, 3), (2, 4), (1, 5), (4, 4), (5, 5)}
    cells["ridge_candidate"] = [
        (lon, lat) in selected for lon, lat in zip(cells.x_bin, cells.y_bin)
    ]
    _, components, _, segments = _extract(cells, grid)

    assert len(components) == 1
    assert components.iloc[0].number_junctions >= 1
    assert len(segments) >= 3


def test_stage5_merge_preserves_junction_without_upstream_assumption() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    theta = np.zeros((7, 7))
    cells = _cells(grid, intensity, theta)
    selected = {(1, 1), (2, 2), (3, 3), (5, 1), (4, 2), (3, 4), (3, 5)}
    cells["ridge_candidate"] = [
        (lon, lat) in selected for lon, lat in zip(cells.x_bin, cells.y_bin)
    ]
    _, components, _, segments = _extract(cells, grid)

    assert components.iloc[0].component_geometry == "junction_network"
    assert segments.component_id.nunique() == 1


def test_stage5_unsupported_gap_is_not_bridged() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    cells = _cells(grid, intensity)
    selected = {(1, 3), (2, 3), (4, 3), (5, 3)}
    cells["ridge_candidate"] = [
        (lon, lat) in selected for lon, lat in zip(cells.x_bin, cells.y_bin)
    ]
    cells.loc[cells.cell_id.eq(3 * 7 + 3), "N_out_move"] = 0
    _, components, _, _ = _extract(cells, grid)

    assert len(components) == 2
    assert components.number_unsupported_interruptions.sum() >= 2


def test_stage5_isolated_one_cell_spike_is_retained_but_smoothing_is_explicit() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, 3] = 20.0
    cells = _cells(grid, intensity)
    raw = _ridge(cells, grid, "raw").set_index("cell_id")
    smoothed = _ridge(cells, grid, "smoothed").set_index("cell_id")
    spike = 3 * 7 + 3

    assert raw.loc[spike, "ridge_candidate"]
    assert smoothed.loc[spike, "S0_field_rate"] < raw.loc[spike, "S0_field_rate"]
    assert smoothed.loc[spike, "C_perp_rate"] < raw.loc[spike, "C_perp_rate"]


def test_stage5_weak_background_ridge_remains_in_unthresholded_candidates() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, :] = 1.2
    result = _ridge(_cells(grid, intensity), grid)

    assert result.loc[
        result.y_bin.eq(3) & result.x_bin.eq(3), "ridge_candidate"
    ].item()


def test_stage5_low_r1_cell_is_flagged_without_removal() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, :] = 10.0
    r1 = np.full((7, 7), 0.9)
    r1[3, 3] = 0.3
    result = _ridge(_cells(grid, intensity, r1=r1), grid).set_index("cell_id")
    center = 3 * 7 + 3

    assert result.loc[center, "ridge_candidate"]
    assert result.loc[center, "orientation_ambiguous_diagnostic"]


def test_stage5_dateline_crossing_branch_uses_periodic_connectivity() -> None:
    grid = GridConfig(
        lon_min=-180,
        lon_max=180,
        lat_min=-1,
        lat_max=1,
        dlon=60,
        dlat=1,
        periodic_longitude=True,
    )
    intensity = np.ones((2, 6))
    cells = _cells(grid, intensity)
    selected = {(5, 0), (0, 0), (1, 0)}
    cells["ridge_candidate"] = [
        (lon, lat) in selected for lon, lat in zip(cells.x_bin, cells.y_bin)
    ]
    _, components, _, _ = _extract(cells, grid)

    assert len(components) == 1
    assert components.iloc[0].n_cells == 3
    assert components.iloc[0].x_span == pytest.approx(120.0)


def test_stage5_physical_sampling_scale_changes_with_zonal_spacing_at_latitude() -> (
    None
):
    grid = GridConfig(
        lon_min=0,
        lon_max=3,
        lat_min=-75,
        lat_max=-30,
        dlon=1,
        dlat=15,
        periodic_longitude=False,
    )
    intensity = np.ones((3, 3))
    result = _ridge(_cells(grid, intensity), grid)
    low_lat = result.loc[result.y_bin.eq(2), "grid_x_scale_length"].median()
    high_lat = result.loc[result.y_bin.eq(0), "grid_x_scale_length"].median()

    assert high_lat < low_lat
    assert result.grid_effective_scale_length.nunique() > 1


def test_stage5_smoothing_never_fills_an_unsupported_gap() -> None:
    values = np.ones((3, 3))
    support = np.ones((3, 3), dtype=bool)
    support[1, 1] = False

    smoothed = support_aware_uniform_3x3(values, support, periodic_x=False)

    assert np.isnan(smoothed[1, 1])


def test_stage5_boundary_aware_retains_available_right_side_candidate() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, 3] = 10.0
    support = np.full((7, 7), 30)
    support[4, 3] = 0
    cells = _cells(grid, intensity, support=support)
    original, _ = transverse_ridge_diagnostics(
        cells,
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="two_sided_only",
    )
    amended, _ = transverse_ridge_diagnostics(
        cells,
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="boundary_aware",
    )
    center = 3 * 7 + 3
    original = original.set_index("cell_id").loc[center]
    amended = amended.set_index("cell_id").loc[center]

    assert not original.ridge_candidate
    assert amended.ridge_candidate
    assert amended.ridge_type == "one_sided"
    assert amended.ridge_evaluability_class == "one_sided_evaluable_right"
    assert amended.missing_side == "left"
    assert amended.future_stage6_flank_observability == "left_flank_not_observable"


def test_stage5_one_sided_candidate_never_fabricates_missing_contrast() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, 3] = 10.0
    support = np.full((7, 7), 30)
    support[4, 3] = 0
    result, _ = transverse_ridge_diagnostics(
        _cells(grid, intensity, support=support),
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="boundary_aware",
    )
    center = result.set_index("cell_id").loc[3 * 7 + 3]

    assert np.isnan(center.S_minus_rate)
    assert np.isnan(center.C_perp_rate)
    assert center.C_perp_one_sided_rate == pytest.approx(
        center.S0_field_rate - center.S_plus_rate
    )


def test_stage5_supported_observed_zero_remains_a_valid_zero() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, 3] = 10.0
    intensity[2, 3] = 0.0
    support = np.full((7, 7), 30)
    support[4, 3] = 0
    result, _ = transverse_ridge_diagnostics(
        _cells(grid, intensity, support=support),
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="boundary_aware",
    )
    center = result.set_index("cell_id").loc[3 * 7 + 3]

    assert np.isfinite(center.S_plus_rate)
    assert center.S_plus_rate < 0.01
    assert center.ridge_candidate_one_sided


def test_stage5_no_transverse_side_evaluable_is_not_a_ridge() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, 3] = 10.0
    support = np.full((7, 7), 30)
    support[2, 3] = 0
    support[4, 3] = 0
    result, _ = transverse_ridge_diagnostics(
        _cells(grid, intensity, support=support),
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="boundary_aware",
    )
    center = result.set_index("cell_id").loc[3 * 7 + 3]

    assert center.ridge_evaluability_class == "no_transverse_side_evaluable"
    assert not center.ridge_candidate
    assert np.isnan(center.C_perp_one_sided_rate)


def test_stage5_boundary_policy_leaves_two_sided_candidates_unchanged() -> None:
    grid = _regular_grid()
    intensity = np.ones((7, 7))
    intensity[3, :] = 10.0
    cells = _cells(grid, intensity)
    original, _ = transverse_ridge_diagnostics(
        cells,
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="two_sided_only",
    )
    amended, _ = transverse_ridge_diagnostics(
        cells,
        grid,
        support_threshold=10,
        field_variant="raw",
        config=FluxConfig(),
        ridge_policy="boundary_aware",
    )

    assert original.ridge_candidate.equals(amended.ridge_candidate_two_sided)
    assert (
        amended.loc[amended.ridge_candidate_two_sided, "ridge_type"]
        .eq("two_sided")
        .all()
    )
