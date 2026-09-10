from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lagrangian_fronts._validation_kernel import (
    _unique_comparison,
    compute_global_gradient_fields,
)
from lagrangian_fronts.config import GridConfig


def _gradient_cells(*, missing_cell: tuple[int, int] | None = None):
    grid = GridConfig(
        lon_min=0.0,
        lon_max=5.0,
        lat_min=-2.5,
        lat_max=2.5,
        dlon=1.0,
        dlat=1.0,
        periodic_longitude=False,
    )
    y_bin, x_bin = np.indices((grid.ny, grid.nx))
    scalar = 1.0 + x_bin + 2.0 * y_bin
    scalar = scalar.astype(float)
    if missing_cell is not None:
        scalar[missing_cell] = np.nan
    cells = pd.DataFrame(
        {
            "cell_id": (y_bin * grid.nx + x_bin).ravel(),
            "x_bin": x_bin.ravel(),
            "y_bin": y_bin.ravel(),
            "x": (grid.x_min + x_bin + 0.5).ravel(),
            "y": (grid.y_min + y_bin + 0.5).ravel(),
            "N_out_move": np.full(scalar.size, 30),
            "U_out_all_magnitude_rate": scalar.ravel(),
            "theta_mu_out": np.full(scalar.size, 90.0),
        }
    )
    return cells, grid


def test_stage7_global_gradient_uses_physical_centered_differences() -> None:
    cells, grid = _gradient_cells()
    result, dataset = compute_global_gradient_fields(cells, grid)
    center = result.loc[(result.x_bin == 2) & (result.y_bin == 2)].iloc[0]

    assert center.dx_method == "dx_centered"
    assert center.dy_method == "dy_centered"
    assert center.dS_dx > 0
    assert center.dS_dy > center.dS_dx
    assert np.isclose(center.G_perp_signed, center.dS_dy)
    assert np.isclose(center.G_parallel_signed, center.dS_dx)
    assert dataset.attrs["stage6_used_in_field_construction"] == "false"


def test_stage7_missing_neighbor_uses_one_sided_difference_without_zero_fill() -> None:
    cells, grid = _gradient_cells(missing_cell=(2, 1))
    result, _ = compute_global_gradient_fields(cells, grid)
    center = result.loc[(result.x_bin == 2) & (result.y_bin == 2)].iloc[0]
    missing = result.loc[(result.x_bin == 1) & (result.y_bin == 2)].iloc[0]

    assert center.dx_method == "dx_one_sided"
    assert np.isfinite(center.dS_dx)
    assert np.isnan(missing.S_flux)
    assert np.isnan(missing.dS_dx)
    assert "flux_undefined" in missing.gradient_quality_flags


def test_stage7_unique_consensus_retains_duplicate_linkage_and_flags_spread() -> None:
    frame = pd.DataFrame(
        {
            "comparison_record_id": ["a", "b"],
            "ridge_cell_id": [7, 7],
            "side": ["left", "left"],
            "ridge_type": ["two_sided", "two_sided"],
            "stage5_missing_side": ["none", "none"],
            "component_id": ["c1", "c1"],
            "segment_id": ["s1", "s2"],
            "section_id": ["x1", "x2"],
            "stage6_persistence": [True, True],
            "nearby_branch_contamination": [False, False],
            "high_local_curvature_turning": [False, False],
            "gradient_observability": ["gradient_observable"] * 2,
            "gradient_sample_class": ["gradient_sample_direct"] * 2,
            "quality_flags": ["", ""],
            "flank_x": [1.0, 1.2],
            "flank_y": [0.0, 0.0],
            "flank_distance_length": [50.0, 170.0],
            "absolute_flux_loss": [2.0, 4.0],
            "relative_flux_loss": [0.2, 0.4],
            "G_perp_at_flank": [0.1, 0.2],
            "abs_G_perp_at_flank": [0.1, 0.2],
            "G_parallel_at_flank": [0.01, 0.02],
            "abs_G_parallel_at_flank": [0.01, 0.02],
            "gradient_magnitude_at_flank": [0.11, 0.22],
            "F_perp_gradient_at_flank": [0.9, 0.9],
            "abs_G_perp_at_core": [0.05, 0.05],
            "flank_to_core_abs_G_perp_ratio": [2.0, 4.0],
            "local_max_abs_G_perp": [0.2, 0.3],
            "distance_to_local_gradient_max_length": [10.0, 20.0],
            "distance_to_local_gradient_max_L_eff": [0.1, 0.2],
            "local_abs_G_perp_percentile": [0.8, 0.9],
            "grid_effective_scale_length": [100.0, 100.0],
            "R1_out_center": [0.9, 0.9],
        }
    )
    config = SimpleNamespace(duplicate_disagreement_grid_scales=1.0)

    result = _unique_comparison(frame, config)

    assert len(result) == 1
    assert result.iloc[0].comparison_record_ids == "a;b"
    assert result.iloc[0].flank_distance_length == 110.0
    assert result.iloc[0].duplicate_flank_disagreement
