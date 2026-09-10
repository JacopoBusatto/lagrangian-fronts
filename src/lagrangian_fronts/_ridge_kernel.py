"""Stage-5 transverse-ridge extraction for coherent flux branches.

This module is deliberately independent of physical-current names and of the
earlier angular-mode graph experiment.  It locates scalar transverse ridges,
then applies only 8-neighbor spatial connectivity to those ridge cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .geometry import (
    NEIGHBOR_OFFSETS_8,
    GeographicGeometry,
    SpatialGeometry,
    _bilinear_supported_sample,
    _grid_array,
    _physical_cell_scales,
    _signed_difference,
    support_aware_uniform_3x3,
)


@dataclass(frozen=True)
class Stage5Fields:
    diagnostics: pd.DataFrame
    dataset: xr.Dataset
    intensity_levels: pd.DataFrame
    ridge_comparisons: pd.DataFrame
    component_members: pd.DataFrame
    components: pd.DataFrame
    segment_members: pd.DataFrame
    segments: pd.DataFrame
    representatives: pd.DataFrame
    summary: dict[str, Any]


def transverse_ridge_diagnostics(
    cells: pd.DataFrame,
    grid: Any,
    *,
    support_threshold: int,
    field_variant: str,
    config: Any,
    geometry: SpatialGeometry | None = None,
    ridge_policy: str = "two_sided_only",
) -> tuple[pd.DataFrame, dict[float, float]]:
    """Calculate physical, support-aware transverse samples for one experiment."""
    if ridge_policy not in {"two_sided_only", "boundary_aware"}:
        raise ValueError(f"Unsupported Stage 5 ridge policy: {ridge_policy}")
    geometry = geometry or GeographicGeometry("WGS84", "km")
    required = {
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
        "delta_theta_mu1_out",
    }
    missing_columns = sorted(required - set(cells.columns))
    if missing_columns:
        raise ValueError(f"Stage 5 cells missing columns: {missing_columns}")
    raw_grid = _grid_array(cells, grid, "U_out_all_magnitude_rate")
    support_grid = (
        _grid_array(cells, grid, "N_out_move") >= support_threshold
    ) & np.isfinite(raw_grid)
    if field_variant == "raw":
        source_grid = np.where(support_grid, raw_grid, np.nan)
    elif field_variant == "smoothed":
        source_grid = support_aware_uniform_3x3(
            raw_grid,
            support_grid,
            periodic_x=grid.periodic_x,
        )
    else:
        raise ValueError(f"Unsupported Stage 5 field variant: {field_variant}")

    output = cells.copy()
    output.insert(0, "support_threshold", support_threshold)
    output.insert(1, "field_variant", field_variant)
    x_scale, y_scale, effective_scale = _physical_cell_scales(
        output, grid, geometry
    )
    output["grid_x_scale_length"] = x_scale
    output["grid_y_scale_length"] = y_scale
    output["grid_effective_scale_length"] = effective_scale
    output["transverse_sampling_distance_length"] = (
        config.transverse_scale_grid * effective_scale
    )
    supported = output.N_out_move.ge(support_threshold)
    orientation_defined = output.theta_mu_out.notna()
    source_value = source_grid[
        output.y_bin.to_numpy(np.int64), output.x_bin.to_numpy(np.int64)
    ]
    output["S0_field_rate"] = source_value
    distance = config.transverse_scale_grid * effective_scale
    theta = output.theta_mu_out.to_numpy(float)
    x = output.x.to_numpy(float)
    y = output.y.to_numpy(float)
    minus_x, minus_y, _ = geometry.forward(x, y, theta - 90.0, distance)
    plus_x, plus_y, _ = geometry.forward(x, y, theta + 90.0, distance)
    output["transverse_minus_x"] = minus_x
    output["transverse_minus_y"] = minus_y
    output["transverse_plus_x"] = plus_x
    output["transverse_plus_y"] = plus_y
    sample_minus, boundary_minus, missing_minus = _bilinear_supported_sample(
        source_grid,
        support_grid,
        minus_x,
        minus_y,
        grid,
        weight_tolerance=config.interpolation_weight_tolerance,
    )
    sample_plus, boundary_plus, missing_plus = _bilinear_supported_sample(
        source_grid,
        support_grid,
        plus_x,
        plus_y,
        grid,
        weight_tolerance=config.interpolation_weight_tolerance,
    )
    output["S_minus_rate"] = sample_minus
    output["S_plus_rate"] = sample_plus
    output["transverse_left_evaluable"] = np.isfinite(sample_minus)
    output["transverse_right_evaluable"] = np.isfinite(sample_plus)
    output["transverse_left_domain_boundary"] = boundary_minus
    output["transverse_right_domain_boundary"] = boundary_plus
    output["transverse_left_missing_support"] = missing_minus
    output["transverse_right_missing_support"] = missing_plus
    left_reason = np.full(len(output), "available", dtype=object)
    right_reason = np.full(len(output), "available", dtype=object)
    left_reason[missing_minus] = "unknown_missing_support"
    right_reason[missing_plus] = "unknown_missing_support"
    left_reason[boundary_minus] = "domain_boundary"
    right_reason[boundary_plus] = "domain_boundary"
    output["transverse_left_status"] = left_reason
    output["transverse_right_status"] = right_reason
    output["D_minus_rate"] = source_value - sample_minus
    output["D_plus_rate"] = source_value - sample_plus
    output["C_perp_rate"] = source_value - (sample_minus + sample_plus) / 2.0
    output["C_perp_normalized"] = np.where(
        source_value != 0,
        output.C_perp_rate / source_value,
        np.nan,
    )
    center_evaluable = supported & orientation_defined & np.isfinite(source_value)
    left_evaluable = np.isfinite(sample_minus)
    right_evaluable = np.isfinite(sample_plus)
    evaluability_class = np.full(
        len(output), "no_transverse_side_evaluable", dtype=object
    )
    evaluability_class[left_evaluable & ~right_evaluable] = "one_sided_evaluable_left"
    evaluability_class[~left_evaluable & right_evaluable] = "one_sided_evaluable_right"
    evaluability_class[left_evaluable & right_evaluable] = "two_sided_evaluable"
    output["ridge_evaluability_class"] = evaluability_class
    missing_side = np.full(len(output), "left_and_right", dtype=object)
    missing_side[left_evaluable & right_evaluable] = "none"
    missing_side[left_evaluable & ~right_evaluable] = "right"
    missing_side[~left_evaluable & right_evaluable] = "left"
    output["missing_side"] = missing_side
    two_sided_evaluable = center_evaluable & left_evaluable & right_evaluable
    one_sided_evaluable = center_evaluable & np.logical_xor(
        left_evaluable, right_evaluable
    )
    at_least_one_side_evaluable = center_evaluable & (left_evaluable | right_evaluable)
    output["ridge_center_evaluable"] = center_evaluable
    output["ridge_two_sided_evaluable"] = two_sided_evaluable
    output["ridge_one_sided_evaluable"] = one_sided_evaluable
    output["ridge_no_transverse_side_evaluable"] = center_evaluable & ~(
        left_evaluable | right_evaluable
    )
    two_sided_candidate = (
        two_sided_evaluable
        & output.D_minus_rate.ge(-config.ridge_comparison_tolerance)
        & output.D_plus_rate.ge(-config.ridge_comparison_tolerance)
    )
    available_difference = np.where(
        left_evaluable & ~right_evaluable,
        output.D_minus_rate,
        np.where(~left_evaluable & right_evaluable, output.D_plus_rate, np.nan),
    )
    output["D_available_rate"] = available_difference
    output["C_perp_one_sided_rate"] = np.where(
        one_sided_evaluable, available_difference, np.nan
    )
    output["C_perp_one_sided_normalized"] = np.where(
        one_sided_evaluable & (source_value != 0),
        output.C_perp_one_sided_rate / source_value,
        np.nan,
    )
    one_sided_candidate = one_sided_evaluable & output.C_perp_one_sided_rate.ge(
        -config.ridge_comparison_tolerance
    )
    output["ridge_candidate_two_sided"] = two_sided_candidate
    output["ridge_candidate_one_sided"] = one_sided_candidate
    output["ridge_candidate_boundary_aware"] = two_sided_candidate | one_sided_candidate
    evaluable = (
        supported
        & orientation_defined
        & np.isfinite(source_value)
        & np.isfinite(sample_minus)
        & np.isfinite(sample_plus)
    )
    output["ridge_evaluable"] = (
        evaluable if ridge_policy == "two_sided_only" else at_least_one_side_evaluable
    )
    output["ridge_candidate"] = (
        two_sided_candidate
        if ridge_policy == "two_sided_only"
        else output.ridge_candidate_boundary_aware
    )
    ridge_type = np.full(len(output), "not_ridge", dtype=object)
    ridge_type[two_sided_candidate] = "two_sided"
    ridge_type[one_sided_candidate] = "one_sided"
    output["ridge_type"] = ridge_type
    observability = np.full(len(output), "not_branch_core", dtype=object)
    observability[two_sided_candidate] = "two_sided_branch_core"
    observability[one_sided_candidate] = "one_sided_branch_core"
    output["branch_core_observability"] = observability
    flank_observability = np.full(len(output), "not_branch_core", dtype=object)
    flank_observability[two_sided_candidate] = "both_sides_observable"
    flank_observability[one_sided_candidate & output.missing_side.eq("left")] = (
        "left_flank_not_observable"
    )
    flank_observability[one_sided_candidate & output.missing_side.eq("right")] = (
        "right_flank_not_observable"
    )
    output["future_stage6_flank_observability"] = flank_observability
    output["ridge_policy"] = ridge_policy
    output["reason_theta_mu_out_undefined"] = supported & ~orientation_defined
    output["reason_transverse_domain_boundary"] = supported & (
        boundary_minus | boundary_plus
    )
    output["reason_transverse_sample_unsupported_or_missing"] = supported & (
        missing_minus | missing_plus
    )
    output["reason_below_operational_support"] = ~supported
    output["reason_interpolation_not_defensible"] = (
        supported
        & orientation_defined
        & (
            ~evaluable
            & ~(boundary_minus | boundary_plus | missing_minus | missing_plus)
        )
    )
    reasons = np.full(len(output), "evaluated_not_transverse_maximum", dtype=object)
    reasons[output.reason_interpolation_not_defensible.to_numpy(bool)] = (
        "interpolation_not_defensible"
    )
    reasons[output.reason_transverse_sample_unsupported_or_missing.to_numpy(bool)] = (
        "transverse_sample_unsupported_or_missing"
    )
    reasons[output.reason_transverse_domain_boundary.to_numpy(bool)] = (
        "analysis_domain_boundary"
    )
    reasons[output.reason_theta_mu_out_undefined.to_numpy(bool)] = (
        "theta_mu_out_undefined"
    )
    reasons[output.reason_below_operational_support.to_numpy(bool)] = (
        "below_operational_support"
    )
    if ridge_policy == "two_sided_only":
        reasons[output.ridge_candidate.to_numpy(bool)] = "transverse_ridge_candidate"
    else:
        reasons[two_sided_candidate.to_numpy(bool)] = (
            "two_sided_transverse_ridge_candidate"
        )
        reasons[one_sided_candidate.to_numpy(bool)] = (
            "one_sided_transverse_ridge_candidate"
        )
    output["ridge_evaluation_reason"] = reasons
    output["orientation_reliable_diagnostic"] = (
        output.R1_out.ge(config.orientation_reliable_R1)
        & output.delta_theta_mu1_out.le(config.direction_disagreement_degrees)
        & orientation_defined
    )
    output["orientation_ambiguous_diagnostic"] = (
        output.R1_out.lt(config.orientation_ambiguous_R1) | ~orientation_defined
    )
    output["low_support_for_experiment"] = ~supported

    population = output.loc[supported & np.isfinite(source_value), "S0_field_rate"]
    thresholds: dict[float, float] = {}
    for quantile in (config.percentile,):
        label = round(100 * quantile)
        threshold_value = float(population.quantile(quantile))
        thresholds[quantile] = threshold_value
        output[f"ridge_candidate_q{label}_two_sided"] = (
            two_sided_candidate & output.S0_field_rate.ge(threshold_value)
        )
        output[f"ridge_candidate_q{label}_one_sided"] = (
            one_sided_candidate & output.S0_field_rate.ge(threshold_value)
        )
        output[f"ridge_candidate_q{label}"] = (
            output.ridge_candidate & output.S0_field_rate.ge(threshold_value)
        )
    return output, thresholds


def _neighbor_ids(cell_id: int, grid: Any) -> list[int]:
    y_bin, x_bin = divmod(int(cell_id), grid.nx)
    neighbors: list[int] = []
    for delta_lat, delta_lon in NEIGHBOR_OFFSETS_8:
        neighbor_lat = y_bin + delta_lat
        neighbor_lon = x_bin + delta_lon
        if neighbor_lat < 0 or neighbor_lat >= grid.ny:
            continue
        if grid.periodic_x:
            neighbor_lon %= grid.nx
        elif neighbor_lon < 0 or neighbor_lon >= grid.nx:
            continue
        neighbors.append(neighbor_lat * grid.nx + neighbor_lon)
    return neighbors


def _connected_components(candidate_ids: set[int], grid: Any) -> list[list[int]]:
    remaining = set(candidate_ids)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in _neighbor_ids(current, grid):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _component_adjacency(component: list[int], grid: Any) -> dict[int, set[int]]:
    selected = set(component)
    return {
        cell_id: set(_neighbor_ids(cell_id, grid)).intersection(selected)
        for cell_id in component
    }


def _ordered_graph_segments(
    component: list[int], grid: Any
) -> tuple[list[tuple[list[int], bool]], dict[int, set[int]]]:
    adjacency = _component_adjacency(component, grid)
    if len(component) == 1:
        return [([component[0]], False)], adjacency
    unused_edges = {
        tuple(sorted((cell_id, neighbor)))
        for cell_id, neighbors in adjacency.items()
        for neighbor in neighbors
        if cell_id != neighbor
    }
    critical = {
        cell_id for cell_id, neighbors in adjacency.items() if len(neighbors) != 2
    }
    segments: list[tuple[list[int], bool]] = []
    for start in sorted(critical):
        for neighbor in sorted(adjacency[start]):
            edge = tuple(sorted((start, neighbor)))
            if edge not in unused_edges:
                continue
            unused_edges.remove(edge)
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in critical:
                next_candidates = sorted(adjacency[current] - {previous})
                if not next_candidates:
                    break
                next_cell = next_candidates[0]
                next_edge = tuple(sorted((current, next_cell)))
                if next_edge not in unused_edges:
                    break
                unused_edges.remove(next_edge)
                path.append(next_cell)
                previous, current = current, next_cell
            segments.append((path, False))
    while unused_edges:
        first_edge = min(unused_edges)
        unused_edges.remove(first_edge)
        start, current = first_edge
        path = [start, current]
        previous = start
        closed = False
        while True:
            options = [
                neighbor
                for neighbor in sorted(adjacency[current] - {previous})
                if tuple(sorted((current, neighbor))) in unused_edges
            ]
            if not options:
                break
            next_cell = options[0]
            unused_edges.remove(tuple(sorted((current, next_cell))))
            if next_cell == start:
                closed = True
                break
            path.append(next_cell)
            previous, current = current, next_cell
        segments.append((path, closed))
    return segments, adjacency


def _minimal_longitude_span(longitudes: np.ndarray) -> float:
    values = np.remainder(np.asarray(longitudes, dtype=float), 360.0)
    if len(values) <= 1:
        return 0.0
    ordered = np.sort(values)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    return float(360.0 - gaps.max())


def _path_geometry(
    path: list[int],
    *,
    closed: bool,
    rows_by_id: pd.DataFrame,
    geometry: SpatialGeometry,
) -> tuple[list[int], np.ndarray, np.ndarray, float, float]:
    def calculate(selected_path: list[int]):
        rows = rows_by_id.loc[selected_path]
        x = rows.x.to_numpy(float)
        y = rows.y.to_numpy(float)
        n_cells = len(selected_path)
        tangent = np.full(n_cells, np.nan, dtype=float)
        edge_lengths: list[float] = []
        edge_pairs = list(zip(range(n_cells - 1), range(1, n_cells)))
        if closed and n_cells > 2:
            edge_pairs.append((n_cells - 1, 0))
        for first, second in edge_pairs:
            bearing, _, distance = geometry.inverse(
                x[first], y[first], x[second], y[second]
            )
            edge_lengths.append(float(distance))
        if n_cells == 1:
            return tangent, np.asarray(edge_lengths), 0.0
        for index in range(n_cells):
            if closed:
                previous = (index - 1) % n_cells
                following = (index + 1) % n_cells
            elif index == 0:
                previous, following = 0, 1
            elif index == n_cells - 1:
                previous, following = n_cells - 2, n_cells - 1
            else:
                previous, following = index - 1, index + 1
            bearing, _, _ = geometry.inverse(
                x[previous], y[previous], x[following], y[following]
            )
            tangent[index] = bearing % 360.0
        theta = rows.theta_mu_out.to_numpy(float)
        agreement = float(np.nansum(np.cos(np.deg2rad(theta - tangent))))
        return tangent, np.asarray(edge_lengths), agreement

    tangent, edge_lengths, agreement = calculate(path)
    if agreement < 0:
        path = list(reversed(path))
        tangent, edge_lengths, agreement = calculate(path)
    rows = rows_by_id.loc[path]
    mismatch = _signed_difference(rows.theta_mu_out.to_numpy(float), tangent)
    length_length = float(edge_lengths.sum())
    if len(path) <= 1 or not len(edge_lengths):
        integrated_ridge_strength = 0.0
    else:
        intensity = rows.S0_field_rate.to_numpy(float)
        if closed:
            paired = (intensity + np.roll(intensity, -1)) / 2.0
        else:
            paired = (intensity[:-1] + intensity[1:]) / 2.0
        integrated_ridge_strength = float(np.sum(paired * edge_lengths))
    return path, tangent, mismatch, length_length, integrated_ridge_strength


def _tangent_turn_metrics(tangent: np.ndarray, *, closed: bool) -> tuple[float, float]:
    finite = np.asarray(tangent, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) <= 1:
        return 0.0, 0.0
    pairs_first = finite
    pairs_second = np.roll(finite, -1)
    if not closed:
        pairs_first = finite[:-1]
        pairs_second = finite[1:]
    turns = np.abs(_signed_difference(pairs_second, pairs_first))
    return float(turns.sum()), float(turns.max(initial=0.0))


def _summary_statistics(
    rows: pd.DataFrame, *, periodic_x: bool
) -> dict[str, Any]:
    statistics = {
        "n_cells": int(rows.cell_id.nunique()),
        "x_span": (
            _minimal_longitude_span(rows.x.to_numpy(float))
            if periodic_x
            else float(rows.x.max() - rows.x.min())
        ),
        "y_span": float(rows.y.max() - rows.y.min()),
        "mean_U_out_all_rate": float(rows.U_out_all_magnitude_rate.mean()),
        "median_U_out_all_rate": float(rows.U_out_all_magnitude_rate.median()),
        "maximum_U_out_all_rate": float(rows.U_out_all_magnitude_rate.max()),
        "mean_C_perp_rate": float(rows.C_perp_rate.mean()),
        "median_C_perp_rate": float(rows.C_perp_rate.median()),
        "mean_C_perp_normalized": float(rows.C_perp_normalized.mean()),
        "mean_R1_out": float(rows.R1_out.mean()),
        "median_R1_out": float(rows.R1_out.median()),
        "mean_R2_out": float(rows.R2_out.mean()),
        "mean_delta_theta_mu1_out": float(rows.delta_theta_mu1_out.mean()),
        "mean_N_out_move": float(rows.N_out_move.mean()),
        "minimum_N_out_move": int(rows.N_out_move.min()),
        "mean_C_neigh_out": float(rows.C_neigh_out_for_experiment.mean()),
        "mean_R1_in": float(rows.R1_in.mean()),
        "mean_N_in_move": float(rows.N_in_move.mean()),
        "mean_abs_delta_theta_io_1": float(rows.delta_theta_io_1.abs().mean()),
        "mean_abs_delta_theta_io_mu": float(rows.delta_theta_io_mu.abs().mean()),
        "n_orientation_reliable_diagnostic": int(
            rows.orientation_reliable_diagnostic.sum()
        ),
        "n_orientation_ambiguous_diagnostic": int(
            rows.orientation_ambiguous_diagnostic.sum()
        ),
    }
    if "ridge_candidate_two_sided" in rows:
        statistics.update(
            {
                "n_two_sided_ridge_cells": int(rows.ridge_candidate_two_sided.sum()),
                "n_one_sided_ridge_cells": int(rows.ridge_candidate_one_sided.sum()),
                "mean_C_perp_one_sided_rate": float(
                    rows.C_perp_one_sided_rate.mean()
                ),
                "median_C_perp_one_sided_rate": float(
                    rows.C_perp_one_sided_rate.median()
                ),
            }
        )
    return statistics


def extract_ridge_components(
    diagnostics: pd.DataFrame,
    grid: Any,
    *,
    support_threshold: int,
    field_variant: str,
    intensity_level: str,
    config: Any,
    geometry: SpatialGeometry | None = None,
    component_id_namespace: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract neutral components and junction-preserving graph segments."""
    candidate_field = (
        "ridge_candidate"
        if intensity_level == "all"
        else f"ridge_candidate_{intensity_level}"
    )
    candidates = diagnostics.loc[diagnostics[candidate_field]].copy()
    geometry = geometry or GeographicGeometry("WGS84", "km")
    candidate_ids = set(candidates.cell_id.astype(int))
    components = _connected_components(candidate_ids, grid)
    rows_by_id = diagnostics.set_index("cell_id", drop=False)
    supported_ids = set(
        diagnostics.loc[diagnostics.N_out_move.ge(support_threshold), "cell_id"].astype(
            int
        )
    )
    component_member_outputs: list[pd.DataFrame] = []
    component_records: list[dict[str, Any]] = []
    segment_member_outputs: list[pd.DataFrame] = []
    segment_records: list[dict[str, Any]] = []
    prefix = f"{component_id_namespace}s{support_threshold}_{field_variant}_{intensity_level}"
    for component_index, component in enumerate(components, start=1):
        component_id = f"{prefix}_c{component_index:04d}"
        graph_segments, adjacency = _ordered_graph_segments(component, grid)
        degree = {cell_id: len(neighbors) for cell_id, neighbors in adjacency.items()}
        endpoints = sorted(cell_id for cell_id, value in degree.items() if value == 1)
        junctions = sorted(cell_id for cell_id, value in degree.items() if value >= 3)
        isolated = sorted(cell_id for cell_id, value in degree.items() if value == 0)
        component_class = (
            "isolated_cell"
            if isolated
            else "junction_network"
            if junctions
            else "loop"
            if not endpoints
            else "simple_path"
        )
        member_rows = rows_by_id.loc[component].copy()
        member_rows.insert(0, "component_id", component_id)
        member_rows.insert(1, "intensity_level", intensity_level)
        member_rows["ridge_graph_degree"] = member_rows.cell_id.map(degree).astype(int)
        member_rows["ridge_graph_endpoint"] = member_rows.ridge_graph_degree.eq(1)
        member_rows["ridge_graph_junction"] = member_rows.ridge_graph_degree.ge(3)
        component_member_outputs.append(member_rows.reset_index(drop=True))
        component_length = 0.0
        component_integrated = 0.0
        component_segment_members: list[pd.DataFrame] = []
        for segment_index, (path, closed) in enumerate(graph_segments, start=1):
            segment_id = f"{component_id}_s{segment_index:04d}"
            path, tangent, mismatch, length_length, integrated = _path_geometry(
                path,
                closed=closed,
                rows_by_id=rows_by_id,
                geometry=geometry,
            )
            segment_rows = rows_by_id.loc[path].copy()
            segment_rows.insert(0, "component_id", component_id)
            segment_rows.insert(1, "segment_id", segment_id)
            segment_rows.insert(2, "intensity_level", intensity_level)
            segment_rows.insert(3, "sequence", np.arange(len(segment_rows)))
            segment_rows["segment_is_closed_loop"] = closed
            segment_rows["ridge_graph_degree"] = segment_rows.cell_id.map(
                degree
            ).astype(int)
            segment_rows["ridge_graph_endpoint"] = segment_rows.ridge_graph_degree.eq(1)
            segment_rows["ridge_graph_junction"] = segment_rows.ridge_graph_degree.ge(3)
            segment_rows["ridge_tangent_bearing"] = tangent
            segment_rows["delta_theta_ridge_mu"] = mismatch
            segment_rows["abs_delta_theta_ridge_mu"] = np.abs(mismatch)
            segment_rows["ridge_vector_abrupt_mismatch_diagnostic"] = (
                segment_rows.abs_delta_theta_ridge_mu.ge(
                    config.abrupt_tangent_mismatch_degrees
                )
            )
            segment_member_outputs.append(segment_rows.reset_index(drop=True))
            component_segment_members.append(segment_rows)
            cumulative_turn, maximum_turn = _tangent_turn_metrics(
                tangent, closed=closed
            )
            segment_stats = _summary_statistics(
                segment_rows, periodic_x=grid.periodic_x
            )
            segment_records.append(
                {
                    "component_id": component_id,
                    "segment_id": segment_id,
                    "support_threshold": support_threshold,
                    "field_variant": field_variant,
                    "intensity_level": intensity_level,
                    "segment_is_closed_loop": closed,
                    "physical_length_length": length_length,
                    "integrated_ridge_strength_area_rate": integrated,
                    "number_endpoints": 0 if closed else min(2, len(path)),
                    "number_junction_endpoints": int(
                        (path[0] in junctions)
                        + (len(path) > 1 and path[-1] in junctions)
                    ),
                    "mean_abs_delta_theta_ridge_mu": float(
                        segment_rows.abs_delta_theta_ridge_mu.mean()
                    ),
                    "median_abs_delta_theta_ridge_mu": float(
                        segment_rows.abs_delta_theta_ridge_mu.median()
                    ),
                    "maximum_abs_delta_theta_ridge_mu": float(
                        segment_rows.abs_delta_theta_ridge_mu.max()
                    ),
                    "n_abrupt_tangent_mismatch_diagnostic": int(
                        segment_rows.ridge_vector_abrupt_mismatch_diagnostic.sum()
                    ),
                    "cumulative_abs_tangent_turn_degrees": cumulative_turn,
                    "maximum_local_tangent_turn_degrees": maximum_turn,
                    **segment_stats,
                }
            )
            component_length += length_length
            component_integrated += integrated
        endpoint_interruptions = sum(
            any(
                neighbor not in supported_ids
                for neighbor in _neighbor_ids(cell_id, grid)
            )
            for cell_id in endpoints
        )
        mismatch_rows = (
            pd.concat(component_segment_members, ignore_index=True)
            if component_segment_members
            else pd.DataFrame()
        )
        component_stats = _summary_statistics(
            member_rows, periodic_x=grid.periodic_x
        )
        component_records.append(
            {
                "component_id": component_id,
                "support_threshold": support_threshold,
                "field_variant": field_variant,
                "intensity_level": intensity_level,
                "component_geometry": component_class,
                "physical_length_length": component_length,
                "integrated_ridge_strength_area_rate": component_integrated,
                "number_segments": len(graph_segments),
                "number_endpoints": len(endpoints),
                "number_junctions": len(junctions),
                "number_unsupported_interruptions": int(endpoint_interruptions),
                "mean_abs_delta_theta_ridge_mu": float(
                    mismatch_rows.abs_delta_theta_ridge_mu.mean()
                )
                if not mismatch_rows.empty
                else np.nan,
                "median_abs_delta_theta_ridge_mu": float(
                    mismatch_rows.abs_delta_theta_ridge_mu.median()
                )
                if not mismatch_rows.empty
                else np.nan,
                "maximum_abs_delta_theta_ridge_mu": float(
                    mismatch_rows.abs_delta_theta_ridge_mu.max()
                )
                if not mismatch_rows.empty
                else np.nan,
                "n_abrupt_tangent_mismatch_diagnostic": int(
                    mismatch_rows.ridge_vector_abrupt_mismatch_diagnostic.sum()
                )
                if not mismatch_rows.empty
                else 0,
                "cumulative_abs_tangent_turn_degrees": float(
                    sum(
                        _tangent_turn_metrics(
                            rows.ridge_tangent_bearing.to_numpy(float),
                            closed=bool(rows.segment_is_closed_loop.iloc[0]),
                        )[0]
                        for rows in component_segment_members
                    )
                ),
                "maximum_local_tangent_turn_degrees": float(
                    max(
                        (
                            _tangent_turn_metrics(
                                rows.ridge_tangent_bearing.to_numpy(float),
                                closed=bool(rows.segment_is_closed_loop.iloc[0]),
                            )[1]
                            for rows in component_segment_members
                        ),
                        default=0.0,
                    )
                ),
                "centroid_y": float(member_rows.y.mean()),
                "centroid_x": (
                    float(
                        np.rad2deg(
                            np.arctan2(
                                np.sin(np.deg2rad(member_rows.x)).mean(),
                                np.cos(np.deg2rad(member_rows.x)).mean(),
                            )
                        )
                    )
                    if geometry.coordinate_system == "geographic"
                    else float(member_rows.x.mean())
                ),
                **component_stats,
            }
        )
    member_columns = [
        "component_id",
        "intensity_level",
        *diagnostics.columns,
        "ridge_graph_degree",
        "ridge_graph_endpoint",
        "ridge_graph_junction",
    ]
    component_members = (
        pd.concat(component_member_outputs, ignore_index=True)
        if component_member_outputs
        else pd.DataFrame(columns=member_columns)
    )
    segment_member_columns = [
        "component_id",
        "segment_id",
        "intensity_level",
        "sequence",
        *diagnostics.columns,
        "segment_is_closed_loop",
        "ridge_graph_degree",
        "ridge_graph_endpoint",
        "ridge_graph_junction",
        "ridge_tangent_bearing",
        "delta_theta_ridge_mu",
        "abs_delta_theta_ridge_mu",
        "ridge_vector_abrupt_mismatch_diagnostic",
    ]
    segment_members = (
        pd.concat(segment_member_outputs, ignore_index=True)
        if segment_member_outputs
        else pd.DataFrame(columns=segment_member_columns)
    )
    component_table = pd.DataFrame.from_records(component_records)
    segment_table = pd.DataFrame.from_records(segment_records)
    if not component_table.empty:
        rank_specs = {
            "rank_physical_length": ("physical_length_length", False),
            "rank_median_flux": ("median_U_out_all_rate", False),
            "rank_integrated_ridge_strength": ("integrated_ridge_strength_area_rate", False),
            "rank_median_ridge_contrast": ("median_C_perp_rate", False),
            "rank_minimum_support": ("minimum_N_out_move", False),
        }
        for rank_name, (field_name, ascending) in rank_specs.items():
            component_table[rank_name] = (
                component_table[field_name]
                .rank(method="min", ascending=ascending, na_option="bottom")
                .astype(int)
            )
    return component_members, component_table, segment_members, segment_table
