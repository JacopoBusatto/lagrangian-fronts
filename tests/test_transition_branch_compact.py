from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from lagrangian_fronts.config import (
    AnalysisConfig,
    GridConfig,
    MatrixConfig,
    OutputConfig,
    SpatialGeometryConfig,
)
from lagrangian_fronts.fronts import _canonical_fronts
from lagrangian_fronts.io import DIRECTIONAL_TABLES, STANDARD_TABLES
from lagrangian_fronts.plotting import (
    CORE_MARKER_STYLES,
    FRONT_MARKER_STYLES,
    _finite_percentile_max,
    _style_quiver_key_label,
)
from lagrangian_fronts.statistics import (
    compact_cell_table,
    compute_transition_statistics,
)


def _config() -> AnalysisConfig:
    return AnalysisConfig(
        matrix=MatrixConfig(path="unused.parquet", id="synthetic", timestep=30.0, time_unit="day", compute=False),
        output=OutputConfig("unused"),
        geometry=SpatialGeometryConfig("geographic", "km", "WGS84"),
        grid=GridConfig(
            lon_min=0.0,
            lon_max=3.0,
            lat_min=0.0,
            lat_max=3.0,
            dlon=1.0,
            dlat=1.0,
            periodic_longitude=False,
        ),
    )


def _four_direction_table() -> pd.DataFrame:
    rows = [(1, 2), (2, 1), (1, 0), (0, 1)]
    return pd.DataFrame(
        {
            "start_lon_bin": pd.Series([1] * 4, dtype="int64"),
            "start_lat_bin": pd.Series([1] * 4, dtype="int64"),
            "end_lon_bin": pd.Series([row[0] for row in rows], dtype="int64"),
            "end_lat_bin": pd.Series([row[1] for row in rows], dtype="int64"),
            "start_lon_center": pd.Series([1.5] * 4, dtype="float64"),
            "start_lat_center": pd.Series([1.5] * 4, dtype="float64"),
            "end_lon_center": pd.Series(
                [row[0] + 0.5 for row in rows], dtype="float64"
            ),
            "end_lat_center": pd.Series(
                [row[1] + 0.5 for row in rows], dtype="float64"
            ),
            "transition_count": pd.Series([5] * 4, dtype="int64"),
            "transition_probability": pd.Series([0.25] * 4, dtype="float64"),
        }
    )


def _eastward_with_stay_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_lon_bin": pd.Series([1, 1], dtype="int64"),
            "start_lat_bin": pd.Series([1, 1], dtype="int64"),
            "end_lon_bin": pd.Series([1, 2], dtype="int64"),
            "end_lat_bin": pd.Series([1, 1], dtype="int64"),
            "start_lon_center": pd.Series([1.5, 1.5], dtype="float64"),
            "start_lat_center": pd.Series([1.5, 1.5], dtype="float64"),
            "end_lon_center": pd.Series([1.5, 2.5], dtype="float64"),
            "end_lat_center": pd.Series([1.5, 1.5], dtype="float64"),
            "transition_count": pd.Series([5, 15], dtype="int64"),
            "transition_probability": pd.Series([0.25, 0.75], dtype="float64"),
        }
    )


def test_compact_workflow_enforces_normalized_source_probability_contract() -> None:
    table = _four_direction_table()
    table.loc[0, "transition_probability"] = 0.20

    with pytest.raises(ValueError, match="row_normalization_failure"):
        compute_transition_statistics(table, _config())


def test_compact_output_preserves_normalized_angular_entropy() -> None:
    statistics = compute_transition_statistics(_four_direction_table(), _config())
    compact = compact_cell_table(statistics.cells)
    source = compact.loc[compact.cell_id.eq(4)].iloc[0]

    assert source.angular_entropy_out == pytest.approx(np.log(4) / np.log(36))
    assert "H_out" not in compact
    assert {"start_x_bin", "start_y_bin", "R1_in", "R2_in"} <= set(compact)


def test_directional_vector_is_distance_free_first_harmonic() -> None:
    statistics = compute_transition_statistics(_eastward_with_stay_table(), _config())
    source = statistics.cells.loc[statistics.cells.cell_id.eq(4)].iloc[0]

    assert source.D_out_move_x == pytest.approx(
        np.sin(np.deg2rad(source.theta1_out))
    )
    assert source.D_out_move_y == pytest.approx(
        np.cos(np.deg2rad(source.theta1_out))
    )
    assert source.D_out_move_magnitude == pytest.approx(source.R1_out)
    assert source.D_out_all_x == pytest.approx(
        source.P_move * source.D_out_move_x
    )
    assert source.D_out_all_y == pytest.approx(
        source.P_move * source.D_out_move_y
    )
    assert source.D_out_all_magnitude == pytest.approx(source.P_move * source.R1_out)
    assert source.theta1_out == pytest.approx(89.986911, abs=1.0e-6)

    identities = statistics.validation_summary["directional_vector_identities"]
    assert identities["D_out_all_equals_P_move_D_out_move_max_abs_x"] == 0.0
    assert identities["D_out_all_equals_P_move_D_out_move_max_abs_y"] == 0.0
    assert identities["D_out_move_magnitude_equals_R1_max_abs"] < 1.0e-15
    assert identities["D_out_all_magnitude_equals_P_move_R1_max_abs"] < 1.0e-15
    assert identities["D_out_move_bearing_equals_theta1_max_abs_degrees"] < 1.0e-12
    assert identities["D_out_all_bearing_equals_theta1_max_abs_degrees"] < 1.0e-12


def test_zero_movement_has_zero_full_directional_vector_but_no_bearing() -> None:
    table = _eastward_with_stay_table().iloc[[0]].copy()
    table["transition_probability"] = 1.0
    moving_source = _eastward_with_stay_table().iloc[[1]].copy()
    moving_source["start_lon_bin"] = 0
    moving_source["start_lat_bin"] = 0
    moving_source["end_lon_bin"] = 1
    moving_source["end_lat_bin"] = 0
    moving_source["start_lon_center"] = 0.5
    moving_source["start_lat_center"] = 0.5
    moving_source["end_lon_center"] = 1.5
    moving_source["end_lat_center"] = 0.5
    moving_source["transition_probability"] = 1.0
    table = pd.concat([table, moving_source], ignore_index=True)
    statistics = compute_transition_statistics(table, _config())
    source = statistics.cells.loc[statistics.cells.cell_id.eq(4)].iloc[0]

    assert source.P_move == 0.0
    assert source.D_out_all_magnitude == 0.0
    assert np.isnan(source.D_out_move_magnitude)
    assert np.isnan(source.theta1_out)


def test_compact_defaults_define_only_one_production_realization() -> None:
    config = _config()

    assert config.statistics.min_moving_support == 10
    assert config.flux.percentile == pytest.approx(0.9)
    assert config.flux.ridge_field == "raw"
    assert config.directional.ridge_comparison_tolerance == pytest.approx(1.0e-12)
    assert config.directional.ridge_plateau_tolerance == pytest.approx(1.0e-12)
    assert config.directional.interpolation_weight_tolerance == pytest.approx(1.0e-10)
    assert config.analysis_version == "5.0.0-production"
    assert not config.run_validation
    assert not config.write_debug_outputs


def test_standard_outputs_and_overlay_labels_are_scientist_facing() -> None:
    assert STANDARD_TABLES == (
        "cell_statistics.parquet",
        "flux_cores.parquet",
        "flux_fronts.parquet",
    )
    assert DIRECTIONAL_TABLES == (
        "directional_corridors.parquet",
        "directional_cores.parquet",
        "directional_fronts.parquet",
        "comparison.parquet",
        "component_comparison.parquet",
    )
    labels = {
        style[2]
        for style in (*CORE_MARKER_STYLES.values(), *FRONT_MARKER_STYLES.values())
    }
    assert labels == {
        "Flux core (two-sided observed)",
        "Flux core (one-sided observed)",
        "Left flux front",
        "Right flux front",
    }


def test_structure_map_colormap_percentile_is_configurable_and_validated() -> None:
    config = _config()

    assert config.plotting.structure_map_max_percentile == 100.0
    assert _finite_percentile_max([0.0, 1.0, 2.0, 100.0, np.nan], 100) == 100.0
    assert _finite_percentile_max([0.0, 1.0, 2.0, 100.0, np.nan], 50) == 1.5
    with pytest.raises(ValueError, match="structure_map_max_percentile"):
        replace(
            config,
            plotting=replace(config.plotting, structure_map_max_percentile=0.0),
        )


def test_quiver_key_label_has_translucent_white_borderless_background() -> None:
    key = SimpleNamespace(text=Text())

    _style_quiver_key_label(key)

    patch = key.text.get_bbox_patch()
    assert patch is not None
    assert patch.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 0.8))
    assert patch.get_edgecolor() == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert patch.get_alpha() == pytest.approx(0.8)


def test_incoming_statistics_are_always_preserved_for_future_topology() -> None:
    statistics = compute_transition_statistics(_four_direction_table(), _config())
    compact = compact_cell_table(statistics.cells)

    assert {
        "R1_in",
        "R2_in",
        "theta1_in_motion_destination",
        "theta_mu_in_motion_destination",
        "delta_theta_mu1_in",
        "delta_theta_io",
    } <= set(compact)


def test_front_product_preserves_observability_and_physical_quantities() -> None:
    cores = pd.DataFrame(
        {
            "cell_id": [10, 11],
            "component_id": ["component_0001", "component_0001"],
            "x": [1.5, 2.5],
            "y": [-40.5, -40.5],
            "ridge_type": ["two_sided", "one_sided"],
            "missing_side": ["none", "right"],
        }
    )
    segment_fronts = pd.DataFrame(
        {
            "cell_id": [10, 10, 11, 11],
            "side": ["left", "left", "left", "right"],
            "candidate_x": [1.0, 1.2, 2.0, 3.0],
            "candidate_y": [-40.0, -40.2, -40.0, -40.0],
            "candidate_distance_length": [50.0, 60.0, 70.0, 80.0],
            "absolute_drop": [2.0, 4.0, 5.0, 9.0],
            "relative_drop": [0.2, 0.4, 0.5, 0.9],
        }
    )

    fronts = _canonical_fronts(cores, segment_fronts)
    left = fronts.loc[fronts.core_cell_id.eq(10) & fronts.side.eq("left")].iloc[0]
    missing = fronts.loc[fronts.core_cell_id.eq(11) & fronts.side.eq("right")].iloc[0]
    no_front = fronts.loc[fronts.core_cell_id.eq(10) & fronts.side.eq("right")].iloc[0]

    assert left.front_x == pytest.approx(1.1)
    assert left.distance_from_core_length == pytest.approx(55.0)
    assert left.flux_loss_rate == pytest.approx(3.0)
    assert left.front_status == "probable_flux_front"
    assert missing.front_status == "side_not_observable"
    assert no_front.front_status == "observable_no_retained_front"
    assert not missing.observable
    assert np.isnan(missing.front_x)
