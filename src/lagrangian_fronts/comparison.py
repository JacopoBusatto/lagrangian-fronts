"""Neutral overlap diagnostics for independent flux/directional selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .cores import CoreSolution
from .directional_cores import DirectionalCoreSolution


@dataclass(frozen=True)
class StructureComparison:
    cells: pd.DataFrame
    components: pd.DataFrame
    summary: dict[str, Any]


def compare_flux_and_directional_structures(
    cells: pd.DataFrame,
    flux: CoreSolution,
    directional: DirectionalCoreSolution,
    config: AnalysisConfig,
) -> StructureComparison:
    """Classify supported cells without matching unlike component geometries."""
    supported = cells.loc[
        cells.N_out_move.ge(config.statistics.min_moving_support),
        [
            "cell_id",
            "x_bin",
            "y_bin",
            "x",
            "y",
            "N_out_move",
            "U_out_all_magnitude_rate",
            "D_out_all_magnitude",
        ],
    ].copy()
    flux_component = flux.cores.set_index("cell_id").component_id
    directional_component = directional.cores.set_index("cell_id").component_id
    supported["flux_component_id"] = supported.cell_id.map(flux_component)
    supported["directional_component_id"] = supported.cell_id.map(
        directional_component
    )
    supported["flux_core"] = supported.flux_component_id.notna()
    supported["directional_core"] = supported.directional_component_id.notna()
    supported["structure_class"] = np.select(
        [
            supported.flux_core & supported.directional_core,
            supported.directional_core,
            supported.flux_core,
        ],
        ["flux_and_directional", "directional_only", "flux_only"],
        default="neither",
    )
    supported = supported.rename(
        columns={"x_bin": "start_x_bin", "y_bin": "start_y_bin"}
    )

    component_records: list[dict[str, Any]] = []
    for structure_type, identifier_field, selected_field, overlap_field in (
        (
            "flux",
            "flux_component_id",
            "flux_core",
            "directional_core",
        ),
        (
            "directional",
            "directional_component_id",
            "directional_core",
            "flux_core",
        ),
    ):
        selected = supported.loc[supported[selected_field]]
        for component_id, group in selected.groupby(identifier_field, sort=True):
            overlap = int(group[overlap_field].sum())
            component_records.append(
                {
                    "structure_type": structure_type,
                    "component_id": component_id,
                    "n_selected_cells": len(group),
                    "n_overlap_cells": overlap,
                    "overlap_fraction": overlap / len(group),
                }
            )
    component_table = pd.DataFrame.from_records(component_records)
    counts = supported.structure_class.value_counts()
    total = len(supported)
    summary = {
        "comparison_population": (
            f"N_out_move >= configured threshold {config.statistics.min_moving_support}"
        ),
        "supported_cells": total,
        "flux_and_directional": int(
            counts.get("flux_and_directional", 0)
        ),
        "directional_only": int(counts.get("directional_only", 0)),
        "flux_only": int(counts.get("flux_only", 0)),
        "neither": int(counts.get("neither", 0)),
        "flux_and_directional_fraction": (
            float(counts.get("flux_and_directional", 0) / total)
            if total
            else np.nan
        ),
        "directional_only_fraction": (
            float(counts.get("directional_only", 0) / total) if total else np.nan
        ),
        "flux_only_fraction": (
            float(counts.get("flux_only", 0) / total) if total else np.nan
        ),
        "neither_fraction": (
            float(counts.get("neither", 0) / total) if total else np.nan
        ),
        "component_matching_performed": False,
    }
    return StructureComparison(supported.reset_index(drop=True), component_table, summary)
