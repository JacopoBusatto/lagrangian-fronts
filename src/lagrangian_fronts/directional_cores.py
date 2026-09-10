"""Transverse ridges of distance-free directional strength."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .directional_corridors import (
    DirectionalCorridorSolution,
    _axial_step_mismatch,
    _compatible_graph,
    _connected_components,
    _minimal_longitude_span,
    _neighbor_ids,
    _quantiles,
)
from .geometry import (
    SpatialGeometry,
    bilinear_supported_sample,
    grid_array,
    make_spatial_geometry,
    physical_cell_scales,
)


@dataclass(frozen=True)
class DirectionalCoreSolution:
    """Ridge-defined directional pathway and its diagnostics."""

    cores: pd.DataFrame
    components: pd.DataFrame
    edges: pd.DataFrame
    candidate_diagnostics: pd.DataFrame
    summary: dict[str, Any]


def _side_status(
    observable: np.ndarray, boundary: np.ndarray, missing: np.ndarray
) -> np.ndarray:
    status = np.full(len(observable), "available", dtype=object)
    status[~observable] = "not_observable"
    status[missing] = "unknown_missing_support"
    status[boundary] = "domain_boundary"
    return status


def _ridge_diagnostics(
    cells: pd.DataFrame,
    candidates: DirectionalCorridorSolution,
    config: AnalysisConfig,
    geometry: SpatialGeometry,
) -> pd.DataFrame:
    """Sample unmasked directional strength across every candidate centre."""
    candidate_ids = set(candidates.corridors.cell_id.astype(int))
    output = cells.loc[cells.cell_id.isin(candidate_ids)].copy()
    candidate_metadata = candidates.corridors.set_index("cell_id")
    for field in (
        "candidate_component_id",
        "candidate_graph_degree",
        "candidate_graph_endpoint",
        "candidate_graph_junction",
    ):
        if field in candidate_metadata:
            output[field] = output.cell_id.map(candidate_metadata[field])

    if output.empty:
        empty_fields: tuple[tuple[str, str], ...] = (
            ("grid_effective_scale_length", "float64"),
            ("transverse_sampling_distance_length", "float64"),
            ("transverse_left_x", "float64"),
            ("transverse_left_y", "float64"),
            ("transverse_right_x", "float64"),
            ("transverse_right_y", "float64"),
            ("transverse_left_strength", "float64"),
            ("transverse_right_strength", "float64"),
            ("left_side_observable", "bool"),
            ("right_side_observable", "bool"),
            ("transverse_left_status", "object"),
            ("transverse_right_status", "object"),
            ("missing_side", "object"),
            ("ridge_type", "object"),
            ("raw_ridge_candidate", "bool"),
        )
        for field, dtype in empty_fields:
            output[field] = pd.Series(dtype=dtype)
        return output

    strength_grid = grid_array(cells, config.grid, "D_out_all_magnitude")
    support_grid = (
        grid_array(cells, config.grid, "N_out_move")
        >= config.statistics.min_moving_support
    ) & np.isfinite(strength_grid)
    x_scale, y_scale, effective_scale = physical_cell_scales(
        output, config.grid, geometry
    )
    distance = config.directional.transverse_scale_grid * effective_scale
    theta = output.theta1_out.to_numpy(float)
    x = output.x.to_numpy(float)
    y = output.y.to_numpy(float)
    left_x, left_y, _ = geometry.forward(x, y, theta - 90.0, distance)
    right_x, right_y, _ = geometry.forward(x, y, theta + 90.0, distance)
    left, left_boundary, left_missing = bilinear_supported_sample(
        strength_grid,
        support_grid,
        left_x,
        left_y,
        config.grid,
        weight_tolerance=config.directional.interpolation_weight_tolerance,
    )
    right, right_boundary, right_missing = bilinear_supported_sample(
        strength_grid,
        support_grid,
        right_x,
        right_y,
        config.grid,
        weight_tolerance=config.directional.interpolation_weight_tolerance,
    )
    left_observable = np.isfinite(left)
    right_observable = np.isfinite(right)
    two_sided = left_observable & right_observable
    one_sided = np.logical_xor(left_observable, right_observable)
    tolerance = config.directional.ridge_comparison_tolerance
    center = output.D_out_all_magnitude.to_numpy(float)
    left_pass = ~left_observable | (center + tolerance >= left)
    right_pass = ~right_observable | (center + tolerance >= right)
    two_sided_ridge = two_sided & left_pass & right_pass
    one_sided_ridge = one_sided & left_pass & right_pass
    raw_ridge = two_sided_ridge | one_sided_ridge

    missing_side = np.full(len(output), "left_and_right", dtype=object)
    missing_side[two_sided] = "none"
    missing_side[left_observable & ~right_observable] = "right"
    missing_side[~left_observable & right_observable] = "left"
    ridge_type = np.full(len(output), "not_ridge", dtype=object)
    ridge_type[~(left_observable | right_observable)] = "not_evaluable"
    ridge_type[two_sided_ridge] = "two_sided"
    ridge_type[one_sided_ridge] = "one_sided"

    output["grid_x_scale_length"] = x_scale
    output["grid_y_scale_length"] = y_scale
    output["grid_effective_scale_length"] = effective_scale
    output["transverse_sampling_distance_length"] = distance
    output["transverse_left_x"] = left_x
    output["transverse_left_y"] = left_y
    output["transverse_right_x"] = right_x
    output["transverse_right_y"] = right_y
    output["transverse_left_strength"] = left
    output["transverse_right_strength"] = right
    output["left_side_observable"] = left_observable
    output["right_side_observable"] = right_observable
    output["transverse_left_status"] = _side_status(
        left_observable, left_boundary, left_missing
    )
    output["transverse_right_status"] = _side_status(
        right_observable, right_boundary, right_missing
    )
    output["missing_side"] = missing_side
    output["ridge_type"] = ridge_type
    output["raw_ridge_candidate"] = raw_ridge
    return output.sort_values("cell_id").reset_index(drop=True)


def _transverse_step(
    first: pd.Series, second: pd.Series, geometry: SpatialGeometry
) -> bool:
    """Return true when the cell connection is primarily cross-stream."""
    forward, backward, _ = geometry.inverse(
        float(first.x), float(first.y), float(second.x), float(second.y)
    )
    first_along = _axial_step_mismatch(float(first.theta1_out), float(forward))
    second_along = _axial_step_mismatch(float(second.theta1_out), float(backward))
    return first_along > 45.0 and second_along > 45.0


def _plateau_components(
    diagnostics: pd.DataFrame,
    config: AnalysisConfig,
    geometry: SpatialGeometry,
) -> tuple[list[list[int]], dict[int, int]]:
    ridge_rows = diagnostics.loc[diagnostics.raw_ridge_candidate].set_index(
        "cell_id", drop=False
    )
    ridge_ids = set(ridge_rows.index.astype(int))
    adjacency = {cell_id: set() for cell_id in sorted(ridge_ids)}
    tolerance = config.directional.ridge_plateau_tolerance
    for first_id in sorted(ridge_ids):
        first = ridge_rows.loc[first_id]
        for second_id in _neighbor_ids(first_id, config.grid):
            if second_id <= first_id or second_id not in ridge_ids:
                continue
            second = ridge_rows.loc[second_id]
            if (
                abs(
                    float(first.D_out_all_magnitude) - float(second.D_out_all_magnitude)
                )
                > tolerance
            ):
                continue
            if not _transverse_step(first, second, geometry):
                continue
            adjacency[first_id].add(second_id)
            adjacency[second_id].add(first_id)
    components = _connected_components(adjacency)
    representative_by_cell: dict[int, int] = {}
    for component in components:
        if len(component) == 1:
            representative_by_cell[component[0]] = component[0]
            continue
        rows = ridge_rows.loc[component]
        x = rows.x.to_numpy(float)
        y = rows.y.to_numpy(float)
        distance_sums: dict[int, float] = {}
        for row in rows.itertuples(index=False):
            _, _, distances = geometry.inverse(
                np.full(len(x), float(row.x)),
                np.full(len(y), float(row.y)),
                x,
                y,
            )
            distance_sums[int(row.cell_id)] = float(np.asarray(distances).sum())
        minimum = min(distance_sums.values())
        central = [
            cell_id
            for cell_id, value in distance_sums.items()
            if np.isclose(value, minimum, rtol=1.0e-12, atol=1.0e-12)
        ]
        support = rows.set_index("cell_id").N_out_move
        representative = min(
            central, key=lambda cell_id: (-float(support.loc[cell_id]), cell_id)
        )
        representative_by_cell.update(
            {cell_id: representative for cell_id in component}
        )
    return components, representative_by_cell


def _add_plateau_diagnostics(
    diagnostics: pd.DataFrame,
    config: AnalysisConfig,
    geometry: SpatialGeometry,
) -> pd.DataFrame:
    output = diagnostics.copy()
    components, representative_by_cell = _plateau_components(output, config, geometry)
    group_by_cell: dict[int, str] = {}
    size_by_cell: dict[int, int] = {}
    for index, component in enumerate(components, start=1):
        group_id = f"directional_plateau_{index:05d}"
        group_by_cell.update({cell_id: group_id for cell_id in component})
        size_by_cell.update({cell_id: len(component) for cell_id in component})
    output["plateau_group_id"] = output.cell_id.map(group_by_cell)
    output["plateau_size"] = output.cell_id.map(size_by_cell).fillna(0).astype(int)
    output["plateau_representative_cell_id"] = output.cell_id.map(
        representative_by_cell
    ).astype("Int64")
    output["plateau_representative"] = [
        bool(
            row.raw_ridge_candidate
            and representative_by_cell.get(int(row.cell_id)) == int(row.cell_id)
        )
        for row in output.itertuples(index=False)
    ]
    return output


def _core_outputs(
    diagnostics: pd.DataFrame,
    config: AnalysisConfig,
    geometry: SpatialGeometry,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[int], int, int]:
    thinned = diagnostics.loc[diagnostics.plateau_representative].copy()
    directional = config.directional
    adjacency, edges = _compatible_graph(
        thinned,
        config.grid,
        geometry,
        maximum_direction_difference=(
            directional.maximum_neighbor_direction_difference_degrees
        ),
        maximum_step_mismatch=directional.maximum_step_direction_mismatch_degrees,
    )
    raw_components = _connected_components(adjacency)
    retained_components = [
        component
        for component in raw_components
        if len(component) >= directional.minimum_component_cells
    ]
    retained_ids = {
        cell_id for component in retained_components for cell_id in component
    }
    edges = edges.loc[
        edges.first_cell_id.isin(retained_ids) & edges.second_cell_id.isin(retained_ids)
    ].copy()

    rows = diagnostics.set_index("cell_id", drop=False)
    member_outputs: list[pd.DataFrame] = []
    component_records: list[dict[str, Any]] = []
    component_by_cell: dict[int, str] = {}
    for index, component in enumerate(retained_components, start=1):
        component_id = f"directional_core_component_{index:04d}"
        component_by_cell.update({cell_id: component_id for cell_id in component})
        component_rows = rows.loc[component].copy()
        degrees = {cell_id: len(adjacency[cell_id]) for cell_id in component}
        component_rows.insert(0, "component_id", component_id)
        component_rows["directional_core_graph_degree"] = component_rows.cell_id.map(
            degrees
        ).astype(int)
        component_rows["directional_core_graph_endpoint"] = (
            component_rows.directional_core_graph_degree.eq(1)
        )
        component_rows["directional_core_graph_junction"] = (
            component_rows.directional_core_graph_degree.ge(3)
        )
        member_outputs.append(component_rows.reset_index(drop=True))
        component_edges = edges.loc[
            edges.first_cell_id.isin(component) & edges.second_cell_id.isin(component)
        ]
        component_records.append(
            {
                "component_id": component_id,
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
                "two_sided_ridge_cells": int(
                    component_rows.ridge_type.eq("two_sided").sum()
                ),
                "one_sided_ridge_cells": int(
                    component_rows.ridge_type.eq("one_sided").sum()
                ),
            }
        )

    members = (
        pd.concat(member_outputs, ignore_index=True)
        if member_outputs
        else pd.DataFrame(
            columns=[
                "component_id",
                *diagnostics.columns,
                "directional_core_graph_degree",
                "directional_core_graph_endpoint",
                "directional_core_graph_junction",
            ]
        )
    )
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
    if not edges.empty:
        edges.insert(0, "component_id", edges.first_cell_id.map(component_by_cell))
    else:
        edges.insert(0, "component_id", pd.Series(dtype="object"))
    discarded_components = sum(
        len(component) < directional.minimum_component_cells
        for component in raw_components
    )
    discarded_cells = sum(
        len(component)
        for component in raw_components
        if len(component) < directional.minimum_component_cells
    )
    return (
        members,
        components,
        edges.reset_index(drop=True),
        retained_ids,
        discarded_components,
        discarded_cells,
    )


def compute_directional_cores(
    cells: pd.DataFrame,
    candidates: DirectionalCorridorSolution,
    config: AnalysisConfig,
) -> DirectionalCoreSolution:
    """Extract, thin, connect, and size-filter directional transverse ridges."""
    geometry = make_spatial_geometry(config.geometry)
    diagnostics = _ridge_diagnostics(cells, candidates, config, geometry)
    diagnostics = _add_plateau_diagnostics(diagnostics, config, geometry)
    (
        members,
        components,
        edges,
        retained_ids,
        discarded_components,
        discarded_cells,
    ) = _core_outputs(diagnostics, config, geometry)
    component_by_cell = (
        members.set_index("cell_id").component_id
        if not members.empty
        else pd.Series(dtype="object")
    )
    diagnostics["directional_core"] = diagnostics.cell_id.isin(retained_ids)
    diagnostics["directional_core_component_id"] = diagnostics.cell_id.map(
        component_by_cell
    )

    core_fields = [
        "component_id",
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
        "ridge_type",
        "missing_side",
        "left_side_observable",
        "right_side_observable",
        "transverse_left_status",
        "transverse_right_status",
        "transverse_left_strength",
        "transverse_right_strength",
        "grid_effective_scale_length",
        "transverse_sampling_distance_length",
        "plateau_group_id",
        "plateau_size",
        "directional_core_graph_degree",
        "directional_core_graph_endpoint",
        "directional_core_graph_junction",
    ]
    cores = members[[name for name in core_fields if name in members]].copy()
    cores = cores.rename(columns={"x_bin": "start_x_bin", "y_bin": "start_y_bin"})

    candidate_fields = [
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
        "grid_effective_scale_length",
        "transverse_sampling_distance_length",
        "transverse_left_strength",
        "transverse_right_strength",
        "left_side_observable",
        "right_side_observable",
        "transverse_left_status",
        "transverse_right_status",
        "missing_side",
        "ridge_type",
        "raw_ridge_candidate",
        "plateau_group_id",
        "plateau_size",
        "plateau_representative_cell_id",
        "plateau_representative",
        "directional_core",
        "directional_core_component_id",
    ]
    candidate_diagnostics = diagnostics[
        [name for name in candidate_fields if name in diagnostics]
    ].copy()
    candidate_diagnostics = candidate_diagnostics.rename(
        columns={"x_bin": "start_x_bin", "y_bin": "start_y_bin"}
    )

    ridge_counts = diagnostics.ridge_type.value_counts()
    summary: dict[str, Any] = {
        "ridge_field": "D_out_all_magnitude",
        "support_threshold": config.statistics.min_moving_support,
        "transverse_scale_grid": config.directional.transverse_scale_grid,
        "ridge_comparison_tolerance": (config.directional.ridge_comparison_tolerance),
        "ridge_plateau_tolerance": config.directional.ridge_plateau_tolerance,
        "interpolation_weight_tolerance": (
            config.directional.interpolation_weight_tolerance
        ),
        "minimum_component_cells": config.directional.minimum_component_cells,
        "directional_candidate_cells": len(diagnostics),
        "raw_two_sided_ridge_cells": int(ridge_counts.get("two_sided", 0)),
        "raw_one_sided_ridge_cells": int(ridge_counts.get("one_sided", 0)),
        "not_ridge_candidate_cells": int(ridge_counts.get("not_ridge", 0)),
        "not_evaluable_candidate_cells": int(ridge_counts.get("not_evaluable", 0)),
        "raw_ridge_cells": int(diagnostics.raw_ridge_candidate.sum()),
        "plateau_groups": int(diagnostics.plateau_group_id.nunique()),
        "plateau_representatives": int(diagnostics.plateau_representative.sum()),
        "directional_core_cells": len(cores),
        "directional_core_components": len(components),
        "discarded_short_core_components": int(discarded_components),
        "discarded_short_core_component_cells": int(discarded_cells),
        **_quantiles(
            cores.D_out_all_magnitude if not cores.empty else pd.Series(dtype=float),
            "core_directional_strength",
        ),
        **_quantiles(
            components.n_cells if not components.empty else pd.Series(dtype=float),
            "core_component_cells",
        ),
    }
    return DirectionalCoreSolution(
        cores=cores.reset_index(drop=True),
        components=components,
        edges=edges,
        candidate_diagnostics=candidate_diagnostics.reset_index(drop=True),
        summary=summary,
    )
