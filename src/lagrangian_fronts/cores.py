"""Boundary-aware current-core detection for one configured realization."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._ridge_kernel import extract_ridge_components, transverse_ridge_diagnostics
from .config import AnalysisConfig
from .geometry import make_spatial_geometry


@dataclass(frozen=True)
class CoreSolution:
    cores: pd.DataFrame
    components: pd.DataFrame
    segment_members: pd.DataFrame
    segments: pd.DataFrame
    threshold_rate: float
    selection_label: str


def compute_current_cores(cells: pd.DataFrame, config: AnalysisConfig) -> CoreSolution:
    percentile = config.flux.percentile
    label = f"q{round(100 * percentile)}"
    branch_config = config.flux
    geometry = make_spatial_geometry(config.geometry)
    diagnostics, levels = transverse_ridge_diagnostics(
        cells,
        config.grid,
        support_threshold=config.statistics.min_moving_support,
        field_variant=config.flux.ridge_field,
        config=branch_config,
        geometry=geometry,
        ridge_policy="boundary_aware",
    )
    neighborhood_field = f"C_neigh_out_1_ge_{config.statistics.min_moving_support}"
    diagnostics["C_neigh_out_for_experiment"] = (
        diagnostics[neighborhood_field]
        if neighborhood_field in diagnostics
        else float("nan")
    )
    members, components, segment_members, segments = extract_ridge_components(
        diagnostics,
        config.grid,
        support_threshold=config.statistics.min_moving_support,
        field_variant=config.flux.ridge_field,
        intensity_level=label,
        config=branch_config,
        geometry=geometry,
    )
    # The frozen extractor returns columnless summaries when no ridge survives.
    # Give this valid empty outcome an explicit structural schema.
    if components.empty:
        components = components.reindex(columns=["component_id"])
    if segments.empty:
        segments = segments.reindex(columns=["component_id", "segment_id"])
    old_components = list(components.component_id)
    component_map = {
        old: f"component_{index:04d}"
        for index, old in enumerate(old_components, start=1)
    }
    old_segments = list(segments.segment_id)
    segment_map = {
        old: f"segment_{index:05d}" for index, old in enumerate(old_segments, start=1)
    }
    for frame in (members, components, segment_members, segments):
        frame["component_id"] = frame.component_id.map(component_map)
        if "segment_id" in frame:
            frame["segment_id"] = frame.segment_id.map(segment_map)
    cores = members[
        [
            "cell_id",
            "x_bin",
            "y_bin",
            "x",
            "y",
            "N_out_move",
            "U_out_all_magnitude_rate",
            "theta_mu_out",
            "R1_out",
            "R2_out",
            "ridge_type",
            "missing_side",
            "component_id",
        ]
    ].copy()
    cores = cores.rename(
        columns={"x_bin": "start_x_bin", "y_bin": "start_y_bin"}
    )
    cores["left_side_observable"] = ~cores.missing_side.isin(["left", "left_and_right"])
    cores["right_side_observable"] = ~cores.missing_side.isin(
        ["right", "left_and_right"]
    )
    return CoreSolution(
        cores,
        components,
        segment_members,
        segments,
        float(levels[percentile]),
        label,
    )
