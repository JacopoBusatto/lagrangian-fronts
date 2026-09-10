from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lagrangian_fronts.comparison import (
    compare_flux_and_directional_structures,
)
from lagrangian_fronts.config import (
    FluxConfig,
    CartesianGridConfig,
    AnalysisConfig,
    DirectionalConfig,
    GridConfig,
    MatrixConfig,
    OutputConfig,
    SpatialGeometryConfig,
)
from lagrangian_fronts.cores import CoreSolution, compute_current_cores
from lagrangian_fronts.directional_cores import compute_directional_cores
from lagrangian_fronts.directional_corridors import (
    compute_directional_corridors,
)
from lagrangian_fronts.directional_fronts import (
    compute_probable_directional_fronts,
)


def _config(nx: int = 7, ny: int = 7) -> AnalysisConfig:
    return AnalysisConfig(
        matrix=MatrixConfig(path="unused.parquet", id="synthetic", timestep=10.0, time_unit="day", compute=False),
        output=OutputConfig("unused"),
        geometry=SpatialGeometryConfig("cartesian", "km"),
        grid=CartesianGridConfig(0.0, float(nx), 0.0, float(ny), 1.0, 1.0),
        directional=DirectionalConfig(
            minimum_P_move=0.5,
            minimum_R1=0.8,
            minimum_strength=0.5,
            maximum_neighbor_direction_difference_degrees=45.0,
            maximum_step_direction_mismatch_degrees=45.0,
            minimum_component_cells=3,
            transverse_scale_grid=1.0,
        ),
    )


def _cells(config: AnalysisConfig) -> pd.DataFrame:
    y_bin, x_bin = np.indices((config.grid.ny, config.grid.nx))
    size = y_bin.size
    theta = np.full(size, 90.0)
    r1 = np.full(size, 0.2)
    p_move = np.ones(size)
    strength = p_move * r1
    return pd.DataFrame(
        {
            "cell_id": (y_bin * config.grid.nx + x_bin).ravel(),
            "x_bin": x_bin.ravel(),
            "y_bin": y_bin.ravel(),
            "x": (config.grid.x_min + (x_bin + 0.5) * config.grid.dx).ravel(),
            "y": (config.grid.y_min + (y_bin + 0.5) * config.grid.dy).ravel(),
            "N_out_move": np.full(size, 20),
            "N_in_move": np.full(size, 20),
            "U_out_all_magnitude_rate": strength.copy(),
            "P_move": p_move,
            "R1_out": r1,
            "R2_out": np.full(size, 0.1),
            "R1_in": np.full(size, 0.9),
            "theta1_out": theta,
            "theta_mu_out": theta.copy(),
            "delta_theta_mu1_out": np.zeros(size),
            "delta_theta_io_1": np.zeros(size),
            "delta_theta_io_mu": np.zeros(size),
            "D_out_all_x": strength * np.sin(np.deg2rad(theta)),
            "D_out_all_y": strength * np.cos(np.deg2rad(theta)),
            "D_out_all_magnitude": strength,
            "C_neigh_out_1_ge_10": np.ones(size),
        }
    )


def _set_directional_cells(
    cells: pd.DataFrame,
    config: AnalysisConfig,
    prescribed: dict[tuple[int, int], float],
    *,
    strength: float = 0.85,
) -> pd.DataFrame:
    output = cells.copy()
    for (x_bin, y_bin), theta in prescribed.items():
        cell_id = y_bin * config.grid.nx + x_bin
        mask = output.cell_id.eq(cell_id)
        output.loc[mask, "R1_out"] = strength
        output.loc[mask, "theta1_out"] = theta
        output.loc[mask, "D_out_all_x"] = strength * np.sin(np.deg2rad(theta))
        output.loc[mask, "D_out_all_y"] = strength * np.cos(np.deg2rad(theta))
        output.loc[mask, "D_out_all_magnitude"] = strength
    return output


def _set_profile(
    cells: pd.DataFrame, config: AnalysisConfig, profile: list[float]
) -> pd.DataFrame:
    output = cells.copy()
    for y_bin, strength in enumerate(profile):
        mask = output.y_bin.eq(y_bin)
        output.loc[mask, "R1_out"] = strength
        output.loc[mask, "D_out_all_magnitude"] = strength
        output.loc[mask, "D_out_all_x"] = strength
        output.loc[mask, "D_out_all_y"] = 0.0
    return output


def _solutions(cells: pd.DataFrame, config: AnalysisConfig):
    candidates = compute_directional_corridors(cells, config)
    cores = compute_directional_cores(cells, candidates, config)
    return candidates, cores


def test_simple_ridge_is_one_cell_wide_and_uses_unmasked_flanks() -> None:
    config = _config()
    config = replace(
        config,
        directional=replace(config.directional, minimum_R1=0.5, minimum_strength=0.5),
    )
    cells = _set_profile(
        cells=_cells(config), config=config, profile=[0.2, 0.4, 0.7, 0.9, 0.7, 0.4, 0.2]
    )
    candidates, cores = _solutions(cells, config)

    assert len(candidates.corridors) == 21
    assert len(cores.cores) == config.grid.nx
    assert cores.cores.start_y_bin.eq(3).all()
    assert cores.cores.ridge_type.eq("two_sided").all()
    np.testing.assert_allclose(cores.cores.transverse_left_strength, 0.7)
    np.testing.assert_allclose(cores.cores.transverse_right_strength, 0.7)


def test_broad_candidate_area_reduces_to_narrow_directional_core() -> None:
    config = _config()
    config = replace(
        config,
        directional=replace(config.directional, minimum_R1=0.5, minimum_strength=0.5),
    )
    cells = _set_profile(_cells(config), config, [0.2, 0.6, 0.75, 0.9, 0.75, 0.6, 0.2])
    candidates, cores = _solutions(cells, config)

    assert len(candidates.corridors) == 5 * config.grid.nx
    assert len(cores.cores) == config.grid.nx
    assert set(cores.cores.start_y_bin) == {3}


def test_flat_transverse_plateau_is_thinned_but_along_stream_is_preserved() -> None:
    config = _config(nx=7, ny=6)
    cells = _set_profile(_cells(config), config, [0.4, 0.7, 0.9, 0.9, 0.7, 0.4])
    _, cores = _solutions(cells, config)
    diagnostics = cores.candidate_diagnostics
    raw_crest = diagnostics.loc[
        diagnostics.raw_ridge_candidate & diagnostics.start_y_bin.isin([2, 3])
    ]

    assert len(raw_crest) == 2 * config.grid.nx
    assert raw_crest.plateau_size.eq(2).all()
    assert len(cores.cores) == config.grid.nx
    assert cores.cores.start_y_bin.eq(2).all()
    assert cores.cores.cell_id.nunique() == config.grid.nx


def test_curved_ridge_follows_local_theta_and_connects() -> None:
    config = _config()
    path = {
        (1, 2): 90.0,
        (2, 2): 90.0,
        (3, 2): 45.0,
        (4, 3): 45.0,
        (5, 4): 45.0,
    }
    cells = _set_directional_cells(_cells(config), config, path, strength=0.9)
    _, cores = _solutions(cells, config)

    assert set(cores.cores.cell_id) == {
        y_bin * config.grid.nx + x_bin for x_bin, y_bin in path
    }
    assert len(cores.components) == 1
    assert cores.components.iloc[0].n_cells == len(path)


def test_opposite_candidates_remain_diagnostic_but_do_not_form_a_core() -> None:
    config = replace(
        _config(),
        directional=replace(_config().directional, minimum_component_cells=2),
    )
    cells = _set_directional_cells(
        _cells(config), config, {(2, 3): 90.0, (3, 3): 270.0}
    )
    candidates, cores = _solutions(cells, config)

    assert len(candidates.corridors) == 2
    assert len(candidates.components) == 2
    assert cores.cores.empty
    assert cores.summary["discarded_short_core_components"] == 2


def test_minimum_component_size_is_applied_after_ridge_extraction() -> None:
    config = _config()
    prescribed = {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    cells.loc[cells.x_bin.eq(3) & cells.y_bin.eq(3), "N_out_move"] = 9
    candidates, cores = _solutions(cells, config)

    assert len(candidates.corridors) == 4
    assert sorted(candidates.components.n_cells) == [2, 2]
    assert cores.cores.empty


def test_directional_core_continues_where_flux_core_is_weak() -> None:
    config = _config(nx=9, ny=9)
    path = {(x_bin, 4): 90.0 for x_bin in range(2, 7)}
    cells = _set_directional_cells(_cells(config), config, path, strength=0.9)
    cells["U_out_all_magnitude_rate"] = 0.1
    cells.loc[cells.y_bin.eq(1), "U_out_all_magnitude_rate"] = 10.0
    cells.loc[
        cells.y_bin.eq(4) & cells.x_bin.between(2, 4),
        "U_out_all_magnitude_rate",
    ] = 10.0
    _, directional = _solutions(cells, config)
    flux = compute_current_cores(cells, config)
    weak_ids = {4 * config.grid.nx + 5, 4 * config.grid.nx + 6}

    assert weak_ids <= set(directional.cores.cell_id)
    assert weak_ids.isdisjoint(set(flux.cores.cell_id))


def test_unrelated_stronger_region_does_not_change_existing_directional_core() -> None:
    config = _config(nx=9, ny=9)
    original_path = {(x_bin, 3): 90.0 for x_bin in range(2, 7)}
    original_cells = _set_directional_cells(
        _cells(config), config, original_path, strength=0.85
    )
    _, original = _solutions(original_cells, config)
    extra_path = {(x_bin, 6): 90.0 for x_bin in range(2, 7)}
    amended_cells = _set_directional_cells(
        original_cells, config, extra_path, strength=0.99
    )
    _, amended = _solutions(amended_cells, config)
    original_ids = {3 * config.grid.nx + x_bin for x_bin in range(2, 7)}

    assert original_ids <= set(original.cores.cell_id)
    assert original_ids <= set(amended.cores.cell_id)


def test_transverse_scale_changes_ridge_membership() -> None:
    config = _config(nx=7, ny=7)
    cells = _set_profile(_cells(config), config, [0.2, 0.9, 0.7, 0.8, 0.7, 0.9, 0.2])
    scale_one = replace(
        config,
        directional=replace(config.directional, transverse_scale_grid=1.0),
    )
    scale_two = replace(
        config,
        directional=replace(config.directional, transverse_scale_grid=2.0),
    )
    _, first = _solutions(cells, scale_one)
    _, second = _solutions(cells, scale_two)
    target_ids = {3 * config.grid.nx + x_bin for x_bin in range(config.grid.nx)}

    assert target_ids <= set(first.cores.cell_id)
    assert target_ids.isdisjoint(set(second.cores.cell_id))


def test_boundary_aware_one_sided_ridge_and_no_side_rejection() -> None:
    config = _config()
    boundary_path = {(x_bin, 0): 90.0 for x_bin in range(1, 6)}
    boundary_cells = _set_directional_cells(
        _cells(config), config, boundary_path, strength=0.9
    )
    _, boundary = _solutions(boundary_cells, config)

    assert len(boundary.cores) == 5
    assert boundary.cores.ridge_type.eq("one_sided").all()
    assert boundary.cores.missing_side.eq("right").all()
    assert boundary.cores.transverse_right_status.eq("domain_boundary").all()

    middle_path = {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    blocked_cells = _set_directional_cells(
        _cells(config), config, middle_path, strength=0.9
    )
    blocked_cells.loc[blocked_cells.y_bin.isin([2, 4]), "N_out_move"] = 0
    _, blocked = _solutions(blocked_cells, config)
    assert blocked.cores.empty
    assert blocked.candidate_diagnostics.ridge_type.eq("not_evaluable").all()


def test_flanks_need_finite_strength_but_not_a_defined_local_angle() -> None:
    config = _config()
    path = {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, path, strength=0.9)
    cells.loc[cells.y_bin.isin([2, 4]), "theta1_out"] = np.nan
    _, cores = _solutions(cells, config)

    assert len(cores.cores) == 5
    assert cores.cores.left_side_observable.all()
    assert cores.cores.right_side_observable.all()
    np.testing.assert_allclose(cores.cores.transverse_left_strength, 0.2)
    np.testing.assert_allclose(cores.cores.transverse_right_strength, 0.2)


def test_directional_core_is_independent_of_branch_config_and_flux_field() -> None:
    config = _config()
    path = {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, path, strength=0.9)
    _, original = _solutions(cells, config)
    amended_config = replace(
        config,
        flux=FluxConfig(
            percentile=0.5,
            transverse_scale_grid=3.0,
            interpolation_weight_tolerance=0.25,
            ridge_comparison_tolerance=0.25,
        ),
    )
    amended_cells = cells.copy()
    amended_cells["U_out_all_magnitude_rate"] = np.linspace(1.0, 100.0, len(cells))
    _, amended = _solutions(amended_cells, amended_config)
    original_fronts = compute_probable_directional_fronts(cells, original, config)
    amended_fronts = compute_probable_directional_fronts(
        amended_cells, amended, amended_config
    )

    assert list(original.cores.cell_id) == list(amended.cores.cell_id)
    pd.testing.assert_series_equal(
        original.cores.D_out_all_magnitude,
        amended.cores.D_out_all_magnitude,
    )
    pd.testing.assert_frame_equal(original_fronts.fronts, amended_fronts.fronts)


def test_core_seeded_fronts_use_core_schema_and_not_all_candidates() -> None:
    config = _config()
    config = replace(
        config,
        directional=replace(config.directional, minimum_R1=0.5, minimum_strength=0.5),
    )
    cells = _set_profile(_cells(config), config, [0.2, 0.6, 0.75, 0.9, 0.75, 0.6, 0.2])
    candidates, cores = _solutions(cells, config)
    fronts = compute_probable_directional_fronts(cells, cores, config)

    assert len(candidates.corridors) > len(cores.cores)
    assert fronts.summary["sections"] == len(cores.cores)
    assert len(fronts.fronts) == 2 * len(cores.cores)
    assert {
        "core_cell_id",
        "core_x",
        "core_y",
        "distance_from_core_axis_length",
    } <= set(fronts.fronts)
    assert not any("corridor" in name for name in fronts.fronts.columns)


def test_cross_section_projects_onto_each_central_local_direction() -> None:
    config = _config()
    prescribed = {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    north = cells.y_bin.eq(4)
    cells.loc[north, "theta1_out"] = 270.0
    cells.loc[north, "D_out_all_x"] = -cells.loc[north, "D_out_all_magnitude"]
    cells.loc[north, "D_out_all_y"] = 0.0
    _, cores = _solutions(cells, config)
    result = compute_probable_directional_fronts(cells, cores, config)
    center_id = 3 * config.grid.nx + 3
    section = result.cross_sections.loc[
        result.cross_sections.core_cell_id.eq(center_id)
    ]
    axis = section.loc[section.offset_index_from_core_cell.eq(0)].iloc[0]
    left = section.loc[section.offset_index_from_core_cell.eq(-1)].iloc[0]

    assert axis.theta1_out_center == 90.0
    assert axis.D_parallel_raw == pytest.approx(0.85)
    assert left.D_parallel_raw < 0.0


def test_missing_support_is_unobservable_and_never_a_directional_drop() -> None:
    config = _config()
    prescribed = {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    cells = _set_directional_cells(_cells(config), config, prescribed)
    north = cells.y_bin.eq(4)
    cells.loc[north, "N_out_move"] = 0
    cells.loc[
        north,
        ["D_out_all_x", "D_out_all_y", "D_out_all_magnitude"],
    ] = np.nan
    _, cores = _solutions(cells, config)
    result = compute_probable_directional_fronts(cells, cores, config)
    left = result.fronts.loc[result.fronts.side.eq("left")]

    assert left.front_status.eq("side_not_observable").all()
    assert not left.front_detected.any()
    assert left.absolute_directional_drop.isna().all()


def test_flux_directional_comparison_uses_only_directional_cores() -> None:
    config = _config()
    config = replace(
        config,
        directional=replace(config.directional, minimum_R1=0.5, minimum_strength=0.5),
    )
    cells = _set_profile(_cells(config), config, [0.2, 0.6, 0.75, 0.9, 0.75, 0.6, 0.2])
    candidates, directional = _solutions(cells, config)
    core_ids = list(directional.cores.cell_id)
    flux_ids = [core_ids[0], core_ids[1], 0]
    flux = CoreSolution(
        cores=pd.DataFrame(
            {
                "cell_id": flux_ids,
                "component_id": ["flux_a", "flux_a", "flux_b"],
            }
        ),
        components=pd.DataFrame(),
        segment_members=pd.DataFrame(),
        segments=pd.DataFrame(),
        threshold_rate=1.0,
        selection_label="q90",
    )
    result = compare_flux_and_directional_structures(
        cells, flux, directional, config
    )

    assert result.summary["flux_and_directional"] == 2
    assert result.summary["directional_only"] == len(core_ids) - 2
    assert result.summary["flux_only"] == 1
    assert int(result.cells.directional_core.sum()) == len(core_ids)
    assert len(candidates.corridors) > int(result.cells.directional_core.sum())
    assert "directional_corridor" not in result.cells
    assert not result.summary["component_matching_performed"]


def test_core_product_has_required_compact_fields() -> None:
    config = _config()
    cells = _set_directional_cells(
        _cells(config), config, {(x_bin, 3): 90.0 for x_bin in range(1, 6)}
    )
    _, cores = _solutions(cells, config)

    assert {
        "component_id",
        "cell_id",
        "start_x_bin",
        "start_y_bin",
        "x",
        "y",
        "N_out_move",
        "P_move",
        "R1_out",
        "theta1_out",
        "D_out_all_x",
        "D_out_all_y",
        "D_out_all_magnitude",
        "ridge_type",
        "missing_side",
        "left_side_observable",
        "right_side_observable",
        "transverse_left_strength",
        "transverse_right_strength",
    } <= set(cores.cores)
    assert "U_out_all_magnitude_rate" not in cores.cores


def test_periodic_longitude_connects_directional_core_across_seam() -> None:
    config = AnalysisConfig(
        matrix=MatrixConfig(path="unused.parquet", id="periodic", timestep=10.0, time_unit="day", compute=False),
        output=OutputConfig("unused"),
        geometry=SpatialGeometryConfig("geographic", "km", "WGS84"),
        grid=GridConfig(0.0, 360.0, -30.0, 30.0, 60.0, 10.0, True),
        directional=DirectionalConfig(minimum_component_cells=3),
    )
    cells = _set_directional_cells(
        _cells(config), config, {(5, 3): 90.0, (0, 3): 90.0, (1, 3): 90.0}
    )
    candidates, cores = _solutions(cells, config)

    assert len(candidates.components) == 1
    assert len(cores.components) == 1
    assert set(cores.cores.start_x_bin) == {0, 1, 5}


def test_empty_directional_products_keep_public_schema() -> None:
    config = _config()
    cells = _cells(config)
    candidates, cores = _solutions(cells, config)
    fronts = compute_probable_directional_fronts(cells, cores, config)

    assert candidates.corridors.empty
    assert cores.cores.empty
    assert {"cell_id", "ridge_type", "transverse_left_strength"} <= set(
        cores.candidate_diagnostics
    )
    assert {"core_cell_id", "core_x", "core_y", "front_status"} <= set(fronts.fronts)


@pytest.mark.parametrize(
    "field",
    [
        "ridge_comparison_tolerance",
        "ridge_plateau_tolerance",
        "interpolation_weight_tolerance",
    ],
)
def test_directional_numerical_tolerances_must_be_nonnegative(field: str) -> None:
    config = _config()
    with pytest.raises(ValueError, match="directional numerical tolerances"):
        replace(
            config,
            directional=replace(config.directional, **{field: -1.0}),
        )
