"""Absolute-threshold candidate areas in the distance-free directional field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .geometry import (
    NEIGHBOR_OFFSETS_8,
    SpatialGeometry,
    make_spatial_geometry,
    signed_angle_difference,
)


@dataclass(frozen=True)
class DirectionalCorridorSolution:
    """Broad candidate-area diagnostic, before transverse-ridge selection."""

    corridors: pd.DataFrame
    components: pd.DataFrame
    edges: pd.DataFrame
    summary: dict[str, Any]


def _neighbor_ids(cell_id: int, grid: Any) -> list[int]:
    y_bin, x_bin = divmod(int(cell_id), grid.nx)
    neighbors: list[int] = []
    for delta_y, delta_x in NEIGHBOR_OFFSETS_8:
        neighbor_y = y_bin + delta_y
        neighbor_x = x_bin + delta_x
        if neighbor_y < 0 or neighbor_y >= grid.ny:
            continue
        if grid.periodic_x:
            neighbor_x %= grid.nx
        elif neighbor_x < 0 or neighbor_x >= grid.nx:
            continue
        neighbors.append(neighbor_y * grid.nx + neighbor_x)
    return neighbors


def _axial_step_mismatch(direction: float, bearing: float) -> float:
    """Difference from an unoriented local connection axis in [0, 90]."""
    forward = abs(float(signed_angle_difference(direction, bearing)))
    backward = abs(float(signed_angle_difference(direction, bearing + 180.0)))
    return min(forward, backward)


def _compatible_graph(
    candidates: pd.DataFrame,
    grid: Any,
    geometry: SpatialGeometry,
    *,
    maximum_direction_difference: float,
    maximum_step_mismatch: float,
) -> tuple[dict[int, set[int]], pd.DataFrame]:
    """Connect locally aligned, direction-compatible candidate cells."""
    rows = candidates.set_index("cell_id", drop=False)
    candidate_ids = set(rows.index.astype(int))
    adjacency = {cell_id: set() for cell_id in sorted(candidate_ids)}
    records: list[dict[str, float | int]] = []
    for first_id in sorted(candidate_ids):
        first = rows.loc[first_id]
        for second_id in _neighbor_ids(first_id, grid):
            if second_id <= first_id or second_id not in candidate_ids:
                continue
            second = rows.loc[second_id]
            direction_difference = abs(
                float(
                    signed_angle_difference(
                        float(second.theta1_out), float(first.theta1_out)
                    )
                )
            )
            if direction_difference > maximum_direction_difference:
                continue
            forward, backward, distance = geometry.inverse(
                float(first.x),
                float(first.y),
                float(second.x),
                float(second.y),
            )
            first_step_mismatch = _axial_step_mismatch(
                float(first.theta1_out), float(forward)
            )
            second_step_mismatch = _axial_step_mismatch(
                float(second.theta1_out), float(backward)
            )
            if max(first_step_mismatch, second_step_mismatch) > maximum_step_mismatch:
                continue
            adjacency[first_id].add(second_id)
            adjacency[second_id].add(first_id)
            records.append(
                {
                    "first_cell_id": first_id,
                    "second_cell_id": second_id,
                    "edge_length_length": float(distance),
                    "direction_difference_degrees": direction_difference,
                    "first_step_mismatch_degrees": first_step_mismatch,
                    "second_step_mismatch_degrees": second_step_mismatch,
                    "maximum_step_mismatch_degrees": max(
                        first_step_mismatch, second_step_mismatch
                    ),
                    "direction_compatibility_cosine": float(
                        np.cos(np.deg2rad(direction_difference))
                    ),
                }
            )
    columns = [
        "first_cell_id",
        "second_cell_id",
        "edge_length_length",
        "direction_difference_degrees",
        "first_step_mismatch_degrees",
        "second_step_mismatch_degrees",
        "maximum_step_mismatch_degrees",
        "direction_compatibility_cosine",
    ]
    return adjacency, pd.DataFrame.from_records(records, columns=columns)


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    remaining = set(adjacency)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            new = sorted(adjacency[current].intersection(remaining), reverse=True)
            remaining.difference_update(new)
            stack.extend(new)
        components.append(sorted(component))
    return components


def _minimal_longitude_span(longitudes: np.ndarray) -> float:
    values = np.remainder(np.asarray(longitudes, dtype=float), 360.0)
    if len(values) <= 1:
        return 0.0
    ordered = np.sort(values)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    return float(360.0 - gaps.max())


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    finite = values.dropna()
    if finite.empty:
        return {f"{prefix}_{name}": np.nan for name in ("q10", "q50", "q90")}
    return {
        f"{prefix}_q10": float(finite.quantile(0.1)),
        f"{prefix}_q50": float(finite.quantile(0.5)),
        f"{prefix}_q90": float(finite.quantile(0.9)),
    }


def _component_outputs(
    candidates: pd.DataFrame,
    adjacency: dict[int, set[int]],
    edges: pd.DataFrame,
    geometry: SpatialGeometry,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach deterministic, unfiltered candidate-component diagnostics."""
    raw_components = _connected_components(adjacency)
    rows = candidates.set_index("cell_id", drop=False)
    member_outputs: list[pd.DataFrame] = []
    component_records: list[dict[str, Any]] = []
    component_by_cell: dict[int, str] = {}
    for index, component in enumerate(raw_components, start=1):
        component_id = f"directional_candidate_component_{index:04d}"
        component_by_cell.update({cell_id: component_id for cell_id in component})
        component_rows = rows.loc[component].copy()
        degrees = {cell_id: len(adjacency[cell_id]) for cell_id in component}
        component_rows.insert(0, "candidate_component_id", component_id)
        component_rows["candidate_graph_degree"] = component_rows.cell_id.map(
            degrees
        ).astype(int)
        component_rows["candidate_graph_endpoint"] = (
            component_rows.candidate_graph_degree.eq(1)
        )
        component_rows["candidate_graph_junction"] = (
            component_rows.candidate_graph_degree.ge(3)
        )
        member_outputs.append(component_rows.reset_index(drop=True))

        component_edges = edges.loc[
            edges.first_cell_id.isin(component) & edges.second_cell_id.isin(component)
        ]
        component_records.append(
            {
                "candidate_component_id": component_id,
                "n_cells": len(component),
                "n_edges": len(component_edges),
                "network_length_length": float(
                    component_edges.edge_length_length.sum()
                ),
                "number_endpoints": int(sum(value == 1 for value in degrees.values())),
                "number_junctions": int(sum(value >= 3 for value in degrees.values())),
                "x_span": (
                    _minimal_longitude_span(component_rows.x.to_numpy(float))
                    if geometry.coordinate_system == "geographic"
                    else float(component_rows.x.max() - component_rows.x.min())
                ),
                "y_span": float(component_rows.y.max() - component_rows.y.min()),
                "centroid_y": float(component_rows.y.mean()),
                "centroid_x": (
                    float(
                        np.rad2deg(
                            np.arctan2(
                                np.sin(np.deg2rad(component_rows.x)).mean(),
                                np.cos(np.deg2rad(component_rows.x)).mean(),
                            )
                        )
                    )
                    if geometry.coordinate_system == "geographic"
                    else float(component_rows.x.mean())
                ),
                "mean_P_move": float(component_rows.P_move.mean()),
                "median_P_move": float(component_rows.P_move.median()),
                "mean_R1_out": float(component_rows.R1_out.mean()),
                "median_R1_out": float(component_rows.R1_out.median()),
                "mean_directional_strength": float(
                    component_rows.D_out_all_magnitude.mean()
                ),
                "median_directional_strength": float(
                    component_rows.D_out_all_magnitude.median()
                ),
                "mean_neighbor_direction_difference_degrees": float(
                    component_edges.direction_difference_degrees.mean()
                )
                if not component_edges.empty
                else np.nan,
                "maximum_neighbor_direction_difference_degrees": float(
                    component_edges.direction_difference_degrees.max()
                )
                if not component_edges.empty
                else np.nan,
                "mean_step_direction_mismatch_degrees": float(
                    component_edges.maximum_step_mismatch_degrees.mean()
                )
                if not component_edges.empty
                else np.nan,
                "maximum_step_direction_mismatch_degrees": float(
                    component_edges.maximum_step_mismatch_degrees.max()
                )
                if not component_edges.empty
                else np.nan,
            }
        )

    members = (
        pd.concat(member_outputs, ignore_index=True)
        if member_outputs
        else pd.DataFrame(
            columns=[
                "candidate_component_id",
                *candidates.columns,
                "candidate_graph_degree",
                "candidate_graph_endpoint",
                "candidate_graph_junction",
            ]
        )
    )
    output_edges = edges.copy()
    if not output_edges.empty:
        output_edges.insert(
            0,
            "candidate_component_id",
            output_edges.first_cell_id.map(component_by_cell),
        )
    else:
        output_edges.insert(0, "candidate_component_id", pd.Series(dtype="object"))
    components = pd.DataFrame.from_records(component_records)
    if not components.empty:
        components["rank_network_length"] = components.network_length_length.rank(
            method="min", ascending=False, na_option="bottom"
        ).astype(int)
        components["rank_median_directional_strength"] = (
            components.median_directional_strength.rank(
                method="min", ascending=False, na_option="bottom"
            ).astype(int)
        )
    return members, components, output_edges


def compute_directional_corridors(
    cells: pd.DataFrame, config: AnalysisConfig
) -> DirectionalCorridorSolution:
    """Select the broad absolute-threshold directional candidate population."""
    required = {
        "cell_id",
        "x_bin",
        "y_bin",
        "x",
        "y",
        "N_out_move",
        "P_move",
        "R1_out",
        "theta1_out",
        "D_out_all_x",
        "D_out_all_y",
        "D_out_all_magnitude",
    }
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"Directional corridor cells missing columns: {missing}")

    directional = config.directional
    supported = cells.N_out_move.ge(config.statistics.min_moving_support)
    organized = (
        supported
        & cells.theta1_out.notna()
        & cells.P_move.ge(directional.minimum_P_move)
        & cells.R1_out.ge(directional.minimum_R1)
        & cells.D_out_all_magnitude.ge(directional.minimum_strength)
    )
    candidates = cells.loc[organized].copy()
    geometry = make_spatial_geometry(config.geometry)
    adjacency, edges = _compatible_graph(
        candidates,
        config.grid,
        geometry,
        maximum_direction_difference=(
            directional.maximum_neighbor_direction_difference_degrees
        ),
        maximum_step_mismatch=directional.maximum_step_direction_mismatch_degrees,
    )
    members, components, edges = _component_outputs(
        candidates, adjacency, edges, geometry
    )

    supported_cells = cells.loc[supported]
    summary: dict[str, Any] = {
        "support_threshold": config.statistics.min_moving_support,
        "minimum_P_move": directional.minimum_P_move,
        "minimum_R1": directional.minimum_R1,
        "minimum_directional_strength": directional.minimum_strength,
        "maximum_neighbor_direction_difference_degrees": (
            directional.maximum_neighbor_direction_difference_degrees
        ),
        "maximum_step_direction_mismatch_degrees": (
            directional.maximum_step_direction_mismatch_degrees
        ),
        "supported_cells": int(supported.sum()),
        "directional_candidate_cells": len(members),
        "directional_candidate_components_unfiltered": len(components),
        **_quantiles(supported_cells.P_move, "supported_P_move"),
        **_quantiles(supported_cells.R1_out, "supported_R1_out"),
        **_quantiles(
            supported_cells.D_out_all_magnitude, "supported_directional_strength"
        ),
        **_quantiles(
            components.n_cells if not components.empty else pd.Series(dtype=float),
            "candidate_component_cells",
        ),
    }
    concise_fields = [
        "candidate_component_id",
        "cell_id",
        "x_bin",
        "y_bin",
        "x",
        "y",
        "N_out_move",
        "P_move",
        "R1_out",
        "theta1_out",
        "D_out_all_x",
        "D_out_all_y",
        "D_out_all_magnitude",
        "candidate_graph_degree",
        "candidate_graph_endpoint",
        "candidate_graph_junction",
    ]
    corridors = members[[name for name in concise_fields if name in members]].copy()
    corridors = corridors.rename(
        columns={"x_bin": "start_x_bin", "y_bin": "start_y_bin"}
    )
    return DirectionalCorridorSolution(
        corridors=corridors,
        components=components,
        edges=edges,
        summary=summary,
    )
