"""Cross-stream profiles and scientist-facing probable flux fronts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._edge_kernel import compute_stage6_fields
from .config import AnalysisConfig
from .cores import CoreSolution
from .geometry import make_spatial_geometry


@dataclass(frozen=True)
class FrontSolution:
    fronts: pd.DataFrame
    candidate_drops: pd.DataFrame
    segment_fronts: pd.DataFrame
    cross_sections: pd.DataFrame
    section_summaries: pd.DataFrame
    experiment_id: str


def _canonical_fronts(
    cores: pd.DataFrame, segment_fronts: pd.DataFrame
) -> pd.DataFrame:
    """Reduce segment-context selections to one outcome per physical core-side."""
    base = cores[
        [
            "cell_id",
            "component_id",
            "x",
            "y",
            "ridge_type",
            "missing_side",
        ]
    ].rename(columns={"cell_id": "core_cell_id", "x": "core_x", "y": "core_y"})
    sides = pd.DataFrame({"side": ["left", "right"]})
    base = base.merge(sides, how="cross")
    base["observable"] = ~(
        base.missing_side.eq(base.side) | base.missing_side.eq("left_and_right")
    )

    selected = segment_fronts.rename(
        columns={
            "cell_id": "core_cell_id",
            "candidate_x": "front_x",
            "candidate_y": "front_y",
            "candidate_distance_length": "distance_from_core_length",
            "absolute_drop": "flux_loss_rate",
            "relative_drop": "relative_flux_loss",
        }
    )
    median_fields = [
        "front_x",
        "front_y",
        "distance_from_core_length",
        "flux_loss_rate",
        "relative_flux_loss",
    ]
    selected = (
        selected.groupby(["core_cell_id", "side"], sort=True)[median_fields]
        .median()
        .reset_index()
    )
    fronts = base.merge(selected, on=["core_cell_id", "side"], how="left")
    fronts["front_detected"] = fronts.front_x.notna() & fronts.observable
    fronts["front_status"] = np.select(
        [~fronts.observable, fronts.front_detected],
        ["side_not_observable", "probable_flux_front"],
        default="observable_no_retained_front",
    )
    fronts.loc[~fronts.front_detected, median_fields] = np.nan
    side_order = pd.Categorical(fronts.side, categories=["left", "right"], ordered=True)
    fronts = fronts.assign(_side_order=side_order).sort_values(
        ["core_cell_id", "_side_order"]
    )
    return fronts.drop(columns=["_side_order", "missing_side"]).reset_index(drop=True)


def compute_probable_fronts(
    cells: pd.DataFrame, cores: CoreSolution, config: AnalysisConfig
) -> FrontSolution:
    support = config.statistics.min_moving_support
    level = cores.selection_label
    if cores.cores.empty:
        columns = {"core_cell_id": "int64", "component_id": "object", "core_x": "float64",
                   "core_y": "float64", "ridge_type": "object", "side": "object",
                   "observable": "bool", "front_detected": "bool", "front_status": "object",
                   "front_x": "float64", "front_y": "float64", "distance_from_core_length": "float64",
                   "flux_loss_rate": "float64", "relative_flux_loss": "float64"}
        empty = pd.DataFrame({key: pd.Series(dtype=dtype) for key, dtype in columns.items()})
        return FrontSolution(empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                             f"s{support}_{config.flux.ridge_field}_{level}")
    result = compute_stage6_fields(
        cells,
        cores.segment_members,
        cores.segments,
        config.grid,
        stage5_config=config.flux,
        config=config.fronts,
        geometry=make_spatial_geometry(config.geometry),
        boundary_aware_branch_cores=True,
        experiments=((support, level),),
        field_variant=config.flux.ridge_field,
    )
    experiment_id = f"s{support}_{config.flux.ridge_field}_{level}"
    return FrontSolution(
        fronts=_canonical_fronts(cores.cores, result.candidate_flank_points),
        candidate_drops=result.candidate_drop_zones,
        segment_fronts=result.candidate_flank_points,
        cross_sections=result.cross_sections,
        section_summaries=result.section_summaries,
        experiment_id=experiment_id,
    )
