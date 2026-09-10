from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lagrangian_fronts import compute_transition_matrix
from lagrangian_fronts.config import GridConfig, SpatialGeometryConfig
from lagrangian_fronts.matrix import validate_matrix


def tracks(x, times=None, y=None, identities=None):
    return pd.DataFrame(
        {
            "trajectory": identities if identities is not None else ["a"] * len(x),
            "time": np.array(
                times if times is not None else np.arange(len(x)), dtype=float
            ),
            "x": np.array(x, dtype=float),
            "y": np.array(y if y is not None else [0.2] * len(x), dtype=float),
        }
    )


@pytest.mark.parametrize(
    "mode,total,counts",
    [("non_overlapping", 2, [1, 1]), ("every_observation", 3, [2, 1])],
)
def test_segment_modes_counts_and_diagnostics(config, mode, total, counts):
    config = replace(config, matrix=replace(config.matrix, segment_mode=mode))
    result = compute_transition_matrix(tracks([0.2, 0.3, 1.2, 1.3, 2.2]), config)
    assert result.transition_table.transition_count.tolist() == counts
    assert result.diagnostics["retained_in_domain_transitions"] == total
    assert result.transition_table.transition_probability.tolist() == [1.0, 1.0]
    diag = result.diagnostics
    assert diag["candidate_transitions"] == sum(
        diag[k]
        for k in (
            "retained_in_domain_transitions",
            "rejected_missing_endpoint",
            "rejected_interpolation_gap",
            "rejected_outside_domain",
        )
    )
    assert 2 * diag["candidate_transitions"] == sum(
        diag[k]
        for k in (
            "exact_endpoints",
            "interpolated_endpoints",
            "missing_endpoints",
            "gap_rejected_endpoints",
        )
    )


def test_interpolation_and_arbitrary_positive_lag(config):
    config = replace(config, matrix=replace(config.matrix, timestep=3.0))
    result = compute_transition_matrix(tracks([0.2, 1.2, 2.2], [0, 2, 4]), config)
    assert result.transition_table.end_x_bin.tolist() == [1]
    assert result.diagnostics["interpolated_endpoints"] == 1
    assert result.diagnostics["source_timestep_seconds"] == 2.0


def test_gap_safeguard_preserves_anchor_and_resumes(config):
    config = replace(
        config, matrix=replace(config.matrix, timestep=1.0, max_interpolation_gap=1.0)
    )
    result = compute_transition_matrix(tracks([0.2] * 6, [0, 1, 5, 6, 7, 8]), config)
    assert result.transition_table.transition_count.tolist() == [4]
    assert result.diagnostics["candidate_transitions"] == 8
    assert result.diagnostics["rejected_interpolation_gap"] == 4
    assert result.diagnostics["gap_rejected_endpoints"] == 6


def test_both_endpoints_domain_rejection_and_stays(config):
    frame = tracks(
        [0.2, 0.3, 0.2, 1.2, 0.2, 4.2, -0.2, 0.2],
        [0, 2] * 4,
        identities=np.repeat(["stay", "move", "exit", "enter"], 2),
    )
    result = compute_transition_matrix(frame, config)
    assert result.transition_table.transition_count.tolist() == [1, 1]
    assert result.transition_table.transition_probability.tolist() == [0.5, 0.5]
    assert result.diagnostics["rejected_outside_domain"] == 2
    assert result.diagnostics["outside_start"] == 1
    assert result.diagnostics["outside_end"] == 1
    assert result.diagnostics["normalization_residual_max_abs"] == 0.0


def test_half_open_boundaries_and_empty_schema(config):
    result = compute_transition_matrix(tracks([0.0, 4.0], [0, 2]), config)
    assert result.transition_table.empty
    assert result.transition_table.transition_count.dtype == "int64"
    assert result.transition_table.transition_probability.dtype == "float64"
    assert result.diagnostics["rejected_outside_domain"] == 1


@pytest.mark.parametrize(
    "bounds,x,expected",
    [((-180.0, 180.0), [179.0, -179.0], 359), ((0.0, 360.0), [359.0, 1.0], 359)],
)
def test_geographic_wrap_interpolation_and_schema(config, bounds, x, expected):
    config = replace(
        config,
        geometry=SpatialGeometryConfig("geographic", "km", "WGS84"),
        grid=GridConfig(*bounds, 0.0, 4.0, 1.0, 1.0, True),
        matrix=replace(config.matrix, timestep=1.0),
    )
    result = compute_transition_matrix(tracks(x, [0, 2]), config)
    assert set(result.transition_table) == {
        f"{side}_{coord}_{suffix}"
        for side in ("start", "end")
        for coord in ("lon", "lat")
        for suffix in ("bin", "center")
    } | {"transition_count", "transition_probability"}
    seam = result.transition_table.loc[
        result.transition_table.start_lon_bin.eq(expected)
    ]
    assert len(seam) == 1
    assert seam.end_lon_bin.iloc[0] == 0
    assert result.transition_table.transition_count.sum() == 2


@pytest.mark.parametrize(
    "bad",
    [
        "duplicates",
        "negative",
        "normalization",
        "count_identity",
        "centres",
        "nonfinite",
    ],
)
def test_invalid_loaded_matrix_contract(config, bad):
    result = compute_transition_matrix(
        tracks([0.2, 0.3, 0.2, 1.2], [0, 2, 0, 2], identities=["a", "a", "b", "b"]),
        config,
    ).transition_table
    if bad == "duplicates":
        result = pd.concat([result, result.iloc[[0]]], ignore_index=True)
    elif bad == "negative":
        result.loc[0, "transition_count"] = -1
    elif bad == "normalization":
        result["transition_probability"] *= 0.5
    elif bad == "count_identity":
        result["transition_probability"] = [0.25, 0.75]
    elif bad == "centres":
        result.loc[0, "start_x_center"] = 0.8
    else:
        result.loc[0, "transition_probability"] = np.nan
    with pytest.raises(ValueError, match="Transition-matrix validation failed"):
        validate_matrix(result, config)
