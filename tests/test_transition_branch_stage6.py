from __future__ import annotations

import numpy as np
import pandas as pd

from lagrangian_fronts._edge_kernel import (
    _contiguous_total_variation,
    compute_stage6_fields,
    robust_contiguous_median,
)
from lagrangian_fronts.config import (
    FluxConfig,
    AnalysisConfig,
    FrontConfig,
    GridConfig,
    MatrixConfig,
    OutputConfig,
    SpatialGeometryConfig,
)
from lagrangian_fronts.cores import compute_current_cores


def _synthetic_stage4() -> tuple[pd.DataFrame, GridConfig]:
    grid = GridConfig(
        lon_min=0.0,
        lon_max=11.0,
        lat_min=-5.5,
        lat_max=5.5,
        dlon=1.0,
        dlat=1.0,
        periodic_longitude=False,
    )
    y_bin, x_bin = np.indices((grid.ny, grid.nx))
    transverse = np.asarray([0.2, 0.5, 1.5, 4.0, 7.0, 10.0, 7.0, 4.0, 1.5, 0.5, 0.2])
    magnitude = np.broadcast_to(transverse[:, None], y_bin.shape).copy()
    support = np.full(y_bin.shape, 30)
    frame = pd.DataFrame(
        {
            "cell_id": (y_bin * grid.nx + x_bin).ravel(),
            "x_bin": x_bin.ravel(),
            "y_bin": y_bin.ravel(),
            "x": (grid.x_min + (x_bin + 0.5) * grid.dlon).ravel(),
            "y": (grid.y_min + (y_bin + 0.5) * grid.dlat).ravel(),
            "N_out_move": support.ravel(),
            "N_in_move": support.ravel(),
            "U_out_all_x_rate": magnitude.ravel(),
            "U_out_all_y_rate": np.zeros(magnitude.size),
            "U_out_all_magnitude_rate": magnitude.ravel(),
            "U_out_move_magnitude_rate": (magnitude / 0.8).ravel(),
            "P_move": np.full(magnitude.size, 0.8),
            "theta_mu_out": np.full(magnitude.size, 90.0),
            "R1_out": np.full(magnitude.size, 0.9),
            "R2_out": np.full(magnitude.size, 0.7),
            "delta_theta_mu1_out": np.full(magnitude.size, 2.0),
            "R1_in": np.full(magnitude.size, 0.85),
            "delta_theta_io_1": np.zeros(magnitude.size),
            "delta_theta_io_mu": np.zeros(magnitude.size),
            "C_neigh_out_1_ge_10": np.full(magnitude.size, 0.9),
            "C_neigh_in_1_ge_10": np.full(magnitude.size, 0.8),
            "C_neigh_out_mu_ge_10": np.full(magnitude.size, 0.88),
            "C_neigh_in_mu_ge_10": np.full(magnitude.size, 0.78),
            "C_neigh_out_1_ge_20": np.full(magnitude.size, 0.9),
            "C_neigh_in_1_ge_20": np.full(magnitude.size, 0.8),
            "C_neigh_out_mu_ge_20": np.full(magnitude.size, 0.88),
            "C_neigh_in_mu_ge_20": np.full(magnitude.size, 0.78),
        }
    )
    return frame, grid


def _synthetic_stage6(*, boundary_aware: bool = False):
    cells, grid = _synthetic_stage4()
    stage5_config = FluxConfig()
    compact_config = AnalysisConfig(
        matrix=MatrixConfig(path="unused.parquet", id="synthetic", timestep=30.0, time_unit="day", compute=False),
        output=OutputConfig("unused"),
        geometry=SpatialGeometryConfig("geographic", "km", "WGS84"),
        grid=grid,
    )
    branches = compute_current_cores(cells, compact_config)
    segment_members = branches.segment_members.copy()
    if boundary_aware:
        segment_members["ridge_type"] = "one_sided"
        segment_members["missing_side"] = "right"
        segment_members["branch_core_observability"] = "one_sided_branch_core"
        segment_members["future_stage6_flank_observability"] = (
            "right_flank_not_observable"
        )
    stage6 = compute_stage6_fields(
        cells,
        segment_members,
        branches.segments,
        grid,
        stage5_config=stage5_config,
        config=FrontConfig(),
        boundary_aware_branch_cores=boundary_aware,
        experiments=((10, "q90"),),
    )
    return stage6


def test_stage6_contiguous_median_does_not_bridge_missing_samples() -> None:
    values = np.asarray([1.0, 9.0, np.nan, 100.0, 102.0])
    result = robust_contiguous_median(values, window=3)

    assert np.isnan(result[2])
    assert result[1] == 5.0
    assert result[3] == 101.0


def test_stage6_total_variation_does_not_cross_a_gap() -> None:
    values = np.asarray([1.0, 3.0, np.nan, 100.0, 104.0])

    assert _contiguous_total_variation(values) == 6.0


def test_stage6_uses_compass_bearing_for_signed_tangent_projection() -> None:
    fields = _synthetic_stage6()
    baseline = fields.cross_sections.loc[
        fields.cross_sections.experiment_id.eq("s10_raw_q90")
    ]

    assert np.allclose(
        baseline.U_parallel_raw.dropna(), baseline.U_out_all_x_rate.dropna()
    )
    assert (baseline.U_parallel_raw.dropna() > 0).all()


def test_stage6_retains_required_section_context_and_sample_provenance() -> None:
    fields = _synthetic_stage6()
    required = {
        "U_parallel_raw",
        "U_parallel_smoothed",
        "U_parallel_relative",
        "U_out_all",
        "U_out_move",
        "P_move",
        "R1_out",
        "R2_out",
        "delta_theta_mu1_out",
        "R1_in",
        "C_neigh_out",
        "C_neigh_in",
        "N_out_move",
        "N_in_move",
        "sample_class",
        "quality_flags",
    }

    assert required <= set(fields.cross_sections)
    assert set(fields.cross_sections.sample_class) <= {
        "direct_grid_information",
        "interpolated",
        "missing_because_of_support",
        "missing_because_of_boundary",
        "missing_invalid",
    }


def test_stage6_core_search_is_local_and_finds_the_synthetic_maximum() -> None:
    fields = _synthetic_stage6()
    baseline = fields.section_summaries.loc[
        fields.section_summaries.experiment_id.eq("s10_raw_q90")
    ]

    assert baseline.d_core_grid_scales.abs().max() <= 1.0
    assert (baseline.d_core_grid_scales == 0).all()
    assert np.allclose(baseline.U_parallel_core, 10.0)


def test_stage6_detects_independent_left_and_right_persistent_weakening() -> None:
    fields = _synthetic_stage6()
    baseline = fields.section_summaries.loc[
        fields.section_summaries.experiment_id.eq("s10_raw_q90")
    ]
    flanks = fields.candidate_flank_points.loc[
        fields.candidate_flank_points.experiment_id.eq("s10_raw_q90")
    ]

    assert baseline.left_drop_detected.all()
    assert baseline.right_drop_detected.all()
    assert set(flanks.side) == {"left", "right"}
    assert flanks.along_branch_persistence.all()


def test_stage6_boundary_aware_core_never_turns_unobservable_flank_into_no_drop() -> (
    None
):
    fields = _synthetic_stage6(boundary_aware=True)
    baseline = fields.section_summaries.loc[
        fields.section_summaries.experiment_id.eq("s10_raw_q90")
    ]
    flanks = fields.candidate_flank_points.loc[
        fields.candidate_flank_points.experiment_id.eq("s10_raw_q90")
    ]

    assert baseline.left_flank_observable.all()
    assert (~baseline.right_flank_observable).all()
    assert baseline.right_flank_status.eq("flank_not_observable").all()
    assert baseline.right_drop_distance_length.isna().all()
    assert baseline.left_drop_detected.all()
    assert (~baseline.no_candidate_drop).all()
    assert baseline.all_observable_flanks_have_candidate_drop.all()
    assert set(flanks.side) == {"left"}
    assert fields.summary["s10_raw_q90_unobservable_flanks"] == len(baseline)


def test_stage6_boundary_aware_samples_retain_values_but_exclude_unobservable_side() -> (
    None
):
    fields = _synthetic_stage6(boundary_aware=True)
    baseline = fields.cross_sections.loc[
        fields.cross_sections.experiment_id.eq("s10_raw_q90")
    ]
    right = baseline.loc[baseline.offset_index_from_stage5_ridge.gt(0)]

    assert right.U_parallel_raw.notna().any()
    assert (~right.flank_observable).all()
    assert (~right.analysis_sample_eligible).all()
    assert right.quality_flags.str.contains("right_flank_not_observable").all()


def test_stage6_composites_do_not_exceed_five_same_segment_sections() -> None:
    fields = _synthetic_stage6()

    assert fields.cross_sections.n_composite_sections.max() <= 5
    assert fields.section_summaries.n_neighbor_sections.max() <= 5


def test_stage6_candidate_outputs_are_points_not_connected_lines() -> None:
    fields = _synthetic_stage6()

    assert {"candidate_x", "candidate_y", "side", "drop_slope"} <= set(
        fields.candidate_flank_points
    )
    assert {"geometry", "front_line_id", "next_flank_point_id"}.isdisjoint(
        fields.candidate_flank_points.columns
    )
    assert fields.summary["continuous_front_lines_created"] is False
    assert fields.summary["stage7_implemented"] is False
