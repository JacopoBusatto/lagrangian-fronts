"""Stage-6 branch-relative cross-stream flux weakening diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._progress import track
from .geometry import (
    GeographicGeometry,
    SpatialGeometry,
    _bilinear_supported_sample,
    _grid_array,
    _physical_cell_scales,
    _signed_difference,
)


@dataclass(frozen=True)
class Stage6Fields:
    cross_sections: pd.DataFrame
    section_summaries: pd.DataFrame
    candidate_drop_zones: pd.DataFrame
    candidate_flank_points: pd.DataFrame
    summary: dict[str, Any]


SAMPLED_FIELDS = (
    "U_out_all_x_rate",
    "U_out_all_y_rate",
    "U_out_all_magnitude_rate",
    "U_out_move_magnitude_rate",
    "P_move",
    "R1_out",
    "R2_out",
    "delta_theta_mu1_out",
    "N_out_move",
    "N_in_move",
)
OPTIONAL_SAMPLED_FIELDS = ("R1_in",)


def _append_flag(flags: list[str], condition: bool, name: str) -> None:
    if condition:
        flags.append(name)


def robust_contiguous_median(values: np.ndarray, *, window: int) -> np.ndarray:
    """Centered rolling median within contiguous finite runs only."""
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return output
    splits = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1)
    half = window // 2
    for run in splits:
        for position, index in enumerate(run):
            selected = run[max(0, position - half) : position + half + 1]
            output[index] = float(np.median(values[selected]))
    return output


def _sample_fields(
    prepared_fields: dict[str, np.ndarray],
    support: np.ndarray,
    grid: Any,
    *,
    target_x: np.ndarray,
    target_y: np.ndarray,
    interpolation_weight_tolerance: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    sampled: dict[str, np.ndarray] = {}
    boundary_any = np.zeros(len(target_x), dtype=bool)
    missing_any = np.zeros(len(target_x), dtype=bool)
    for field_name, values in prepared_fields.items():
        field_values, boundary, missing = _bilinear_supported_sample(
            values,
            support,
            target_x,
            target_y,
            grid,
            weight_tolerance=interpolation_weight_tolerance,
        )
        sampled[field_name] = field_values
        if field_name in {
            "U_out_all_x_rate",
            "U_out_all_y_rate",
        }:
            boundary_any |= boundary
            missing_any |= missing
    return sampled, boundary_any, missing_any


def _nearby_branch_contamination(
    center: pd.Series,
    experiment_members: pd.DataFrame,
    geometry: SpatialGeometry,
    config: Any,
) -> bool:
    others = experiment_members.loc[
        experiment_members.cell_id.ne(center.cell_id)
        & experiment_members.component_id.ne(center.component_id)
    ].drop_duplicates("cell_id")
    if others.empty:
        return False
    bearing, _, distance = geometry.inverse(
        np.full(len(others), center.x),
        np.full(len(others), center.y),
        others.x.to_numpy(float),
        others.y.to_numpy(float),
    )
    angle = np.deg2rad(
        _signed_difference(
            np.asarray(bearing), np.full(len(others), center.theta_mu_out)
        )
    )
    distance_length = np.asarray(distance)
    along = distance_length * np.cos(angle)
    cross = distance_length * np.sin(angle)
    scale = float(center.grid_effective_scale_length)
    nearby = (np.abs(along) <= config.nearby_branch_along_distance_scales * scale) & (
        np.abs(cross) <= config.nearby_branch_cross_distance_scales * scale
    )
    return bool(nearby.any())


def _section_rows(
    center: pd.Series,
    experiment_members: pd.DataFrame,
    grid: Any,
    *,
    experiment_id: str,
    config: Any,
    geometry: SpatialGeometry,
    interpolation_weight_tolerance: float,
    prepared_fields: dict[str, np.ndarray],
    support_grid: np.ndarray,
    boundary_aware_branch_cores: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    half_width = config.half_width_grid_scales
    interval = config.sampling_interval_grid_scales
    offsets = np.arange(-half_width, half_width + interval / 2, interval)
    distance_length = offsets * float(center.grid_effective_scale_length)
    bearings = np.where(
        distance_length < 0, center.theta_mu_out - 90.0, center.theta_mu_out + 90.0
    )
    target_x, target_y, _ = geometry.forward(
        np.full(len(offsets), center.x),
        np.full(len(offsets), center.y),
        bearings,
        np.abs(distance_length),
    )
    zero = np.isclose(offsets, 0.0)
    target_x[zero] = center.x
    target_y[zero] = center.y
    sampled, boundary, missing = _sample_fields(
        prepared_fields,
        support_grid,
        grid,
        target_x=target_x,
        target_y=target_y,
        interpolation_weight_tolerance=interpolation_weight_tolerance,
    )
    tangent_east = np.sin(np.deg2rad(center.theta_mu_out))
    tangent_north = np.cos(np.deg2rad(center.theta_mu_out))
    u_parallel = (
        sampled["U_out_all_x_rate"] * tangent_east
        + sampled["U_out_all_y_rate"] * tangent_north
    )
    ridge_type = (
        str(center.get("ridge_type", "two_sided"))
        if boundary_aware_branch_cores
        else "two_sided"
    )
    missing_side = (
        str(center.get("missing_side", "none"))
        if boundary_aware_branch_cores
        else "none"
    )
    left_flank_observable = missing_side not in {"left", "left_and_right"}
    right_flank_observable = missing_side not in {"right", "left_and_right"}
    raw_side = np.where(offsets < 0, "left", np.where(offsets > 0, "right", "core"))
    sample_flank_observable = (
        (raw_side == "core")
        | ((raw_side == "left") & left_flank_observable)
        | ((raw_side == "right") & right_flank_observable)
    )
    valid = np.isfinite(u_parallel)
    allowed_core = (
        valid
        & sample_flank_observable
        & (np.abs(offsets) <= config.core_refinement_grid_scales + 1.0e-12)
    )
    if allowed_core.any():
        allowed_indexes = np.flatnonzero(allowed_core)
        core_index = allowed_indexes[np.nanargmax(u_parallel[allowed_core])]
        core_offset = float(offsets[core_index])
        core_value = float(u_parallel[core_index])
        refined_x = float(target_x[core_index])
        refined_y = float(target_y[core_index])
        core_uncertain = bool(
            np.isclose(abs(core_offset), config.core_refinement_grid_scales)
        )
    else:
        core_index = int(np.flatnonzero(zero)[0])
        core_offset = np.nan
        core_value = np.nan
        refined_x = np.nan
        refined_y = np.nan
        core_uncertain = True
    offset_refined = (
        offsets - core_offset
        if np.isfinite(core_offset)
        else np.full_like(offsets, np.nan)
    )
    distance_refined = offset_refined * float(center.grid_effective_scale_length)
    smooth = robust_contiguous_median(
        u_parallel, window=config.robust_median_window_samples
    )
    relative = (
        u_parallel / core_value if core_value > 0 else np.full_like(u_parallel, np.nan)
    )
    sample_class = np.full(len(offsets), "interpolated", dtype=object)
    sample_class[zero & valid] = "direct_grid_information"
    sample_class[missing] = "missing_because_of_support"
    sample_class[boundary] = "missing_because_of_boundary"
    sample_class[~valid & ~(missing | boundary)] = "missing_invalid"
    nearby = _nearby_branch_contamination(center, experiment_members, geometry, config)
    pixel_junction = bool(center.ridge_graph_junction)
    segment_endpoint = bool(center.ridge_graph_endpoint)
    high_curvature = bool(
        center.maximum_local_tangent_turn_degrees
        >= config.diagnostic_high_curvature_degrees
    )
    section_flags: list[str] = []
    _append_flag(
        section_flags,
        center.N_out_move < center.support_threshold,
        "insufficient_outgoing_support",
    )
    _append_flag(
        section_flags,
        center.N_in_move < center.support_threshold,
        "insufficient_incoming_support",
    )
    _append_flag(
        section_flags, not np.isfinite(center.theta_mu_out), "undefined_theta_mu_out"
    )
    _append_flag(section_flags, center.R1_out < config.diagnostic_low_R1, "low_R1_out")
    _append_flag(
        section_flags,
        center.delta_theta_mu1_out
        > config.diagnostic_large_direction_disagreement_degrees,
        "large_delta_theta_mu1_out",
    )
    _append_flag(section_flags, bool(missing.any()), "unsupported_transverse_sample")
    _append_flag(section_flags, bool(boundary.any()), "domain_boundary")
    _append_flag(
        section_flags,
        not left_flank_observable,
        "left_flank_not_observable",
    )
    _append_flag(
        section_flags,
        not right_flank_observable,
        "right_flank_not_observable",
    )
    _append_flag(section_flags, core_uncertain, "core_location_uncertain")
    _append_flag(section_flags, nearby, "nearby_branch_contamination")
    _append_flag(section_flags, pixel_junction, "pixel_junction_nearby")
    _append_flag(section_flags, segment_endpoint, "segment_endpoint")
    _append_flag(section_flags, high_curvature, "high_local_curvature_turning")
    _append_flag(
        section_flags,
        int(valid.sum()) < config.diagnostic_min_full_section_valid_samples,
        "short_available_cross_section",
    )
    section_id = (
        f"{experiment_id}_{center.segment_id}_cell{int(center.cell_id):05d}"
        f"_seq{int(center.sequence):04d}"
    )
    rows = pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "support_threshold": int(center.support_threshold),
            "intensity_level": center.intensity_level,
            "component_id": center.component_id,
            "segment_id": center.segment_id,
            "section_id": section_id,
            "section_sequence": int(center.sequence),
            "cell_id": int(center.cell_id),
            "ridge_x": float(center.x),
            "ridge_y": float(center.y),
            "refined_core_x": refined_x,
            "refined_core_y": refined_y,
            "offset_index_from_stage5_ridge": offsets,
            "offset_index_from_refined_core": offset_refined,
            "d_from_stage5_ridge_length": distance_length,
            "d_from_refined_core_length": distance_refined,
            "sample_x": target_x,
            "sample_y": target_y,
            "side": np.where(
                offset_refined < 0,
                "left",
                np.where(offset_refined > 0, "right", "core"),
            ),
            "theta_mu_out_center": float(center.theta_mu_out),
            "U_parallel_raw": u_parallel,
            "U_parallel_smoothed": smooth,
            "U_parallel_relative": relative,
            "sample_valid": valid,
            "flank_observable": sample_flank_observable,
            "analysis_sample_eligible": valid & sample_flank_observable,
            "sample_interpolated": sample_class == "interpolated",
            "sample_class": sample_class,
            "ridge_type_center": ridge_type,
            "stage5_missing_side": missing_side,
            "branch_core_observability": (
                str(center.get("branch_core_observability", "two_sided_branch_core"))
                if boundary_aware_branch_cores
                else "two_sided_branch_core"
            ),
            "quality_flags": ";".join(section_flags),
        }
    )
    for field_name, values in sampled.items():
        output_name = {
            "U_out_all_magnitude_rate": "U_out_all",
            "U_out_move_magnitude_rate": "U_out_move",
        }.get(field_name, field_name)
        rows[output_name] = values
    summary = {
        "experiment_id": experiment_id,
        "support_threshold": int(center.support_threshold),
        "intensity_level": center.intensity_level,
        "component_id": center.component_id,
        "segment_id": center.segment_id,
        "section_id": section_id,
        "section_sequence": int(center.sequence),
        "cell_id": int(center.cell_id),
        "ridge_x": float(center.x),
        "ridge_y": float(center.y),
        "theta_mu_out_center": float(center.theta_mu_out),
        "grid_effective_scale_length": float(center.grid_effective_scale_length),
        "d_core_grid_scales": core_offset,
        "d_core_length": core_offset * float(center.grid_effective_scale_length),
        "refined_core_x": refined_x,
        "refined_core_y": refined_y,
        "U_parallel_core": core_value,
        "core_location_uncertain": core_uncertain,
        "nearby_branch_contamination": nearby,
        "pixel_junction_nearby": pixel_junction,
        "segment_endpoint": segment_endpoint,
        "high_local_curvature_turning": high_curvature,
        "low_R1_out": bool(center.R1_out < config.diagnostic_low_R1),
        "R1_out_center": float(center.R1_out),
        "R2_out_center": float(center.R2_out),
        "delta_theta_mu1_out_center": float(center.delta_theta_mu1_out),
        "N_out_move_center": int(center.N_out_move),
        "N_in_move_center": int(center.N_in_move),
        "ridge_type": ridge_type,
        "stage5_missing_side": missing_side,
        "branch_core_observability": (
            str(center.get("branch_core_observability", "two_sided_branch_core"))
            if boundary_aware_branch_cores
            else "two_sided_branch_core"
        ),
        "left_flank_observable": left_flank_observable,
        "right_flank_observable": right_flank_observable,
        "n_observable_flanks": int(left_flank_observable) + int(right_flank_observable),
        "n_unobservable_flanks": 2
        - int(left_flank_observable)
        - int(right_flank_observable),
        "n_valid_samples": int(valid.sum()),
        "n_missing_support_samples": int(missing.sum()),
        "n_missing_boundary_samples": int(boundary.sum()),
        "short_available_cross_section": bool(
            valid.sum() < config.diagnostic_min_full_section_valid_samples
        ),
        "opposing_outer_flux": bool(
            np.any(u_parallel[np.abs(offset_refined) >= 2] < 0)
        )
        if np.isfinite(core_offset)
        else False,
        "profile_total_variation_raw": _contiguous_total_variation(u_parallel),
        "profile_total_variation_smoothed": _contiguous_total_variation(smooth),
        "quality_flags": ";".join(section_flags),
    }
    return rows, summary


def _contiguous_total_variation(values: np.ndarray) -> float:
    """Total variation from adjacent supported samples, never across a gap."""
    values = np.asarray(values, dtype=float)
    pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    return float(np.abs(np.diff(values)[pairs]).sum())


def _add_along_branch_composites(
    cross_sections: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    half_window: int,
    geometry: SpatialGeometry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = cross_sections.copy()
    for field_name in (
        "U_parallel_composite_median",
        "U_parallel_composite_q25",
        "U_parallel_composite_q75",
        "n_composite_sections",
        "n_sections_with_outward_decline",
        "fraction_sections_with_outward_decline",
    ):
        output[field_name] = np.nan
    span_by_section: dict[str, float] = {}
    groups = summaries.groupby(["experiment_id", "segment_id"], sort=False)
    for (experiment_id, segment_id), group_summary in track(
        groups, desc="Flux fronts: profile composites", total=groups.ngroups,
        unit="segments",
    ):
        for focal in group_summary.itertuples(index=False):
            neighbors = group_summary.loc[
                group_summary.section_sequence.between(
                    focal.section_sequence - half_window,
                    focal.section_sequence + half_window,
                )
            ]
            neighbor_ids = set(neighbors.section_id)
            profiles = output.loc[
                output.section_id.isin(neighbor_ids),
                ["offset_index_from_refined_core", "U_parallel_raw", "section_id"],
            ]
            aggregate = profiles.groupby(
                "offset_index_from_refined_core"
            ).U_parallel_raw.agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
                count="count",
            )
            mask = output.section_id.eq(focal.section_id)
            indexes = output.loc[mask, "offset_index_from_refined_core"]
            output.loc[mask, "U_parallel_composite_median"] = indexes.map(
                aggregate["median"]
            ).to_numpy()
            output.loc[mask, "U_parallel_composite_q25"] = indexes.map(
                aggregate["q25"]
            ).to_numpy()
            output.loc[mask, "U_parallel_composite_q75"] = indexes.map(
                aggregate["q75"]
            ).to_numpy()
            output.loc[mask, "n_composite_sections"] = indexes.map(
                aggregate["count"]
            ).to_numpy()
            pivot = profiles.pivot_table(
                index="section_id",
                columns="offset_index_from_refined_core",
                values="U_parallel_raw",
                aggfunc="first",
            )
            focal_rows = output.loc[mask].copy()
            for side in ("left", "right"):
                side_rows = focal_rows.loc[focal_rows.side.isin(("core", side))].copy()
                side_rows["outward_distance"] = side_rows[
                    "offset_index_from_refined_core"
                ].abs()
                side_rows = side_rows.sort_values("outward_distance")
                for pair_index in range(len(side_rows) - 1):
                    inner_offset = side_rows.iloc[
                        pair_index
                    ].offset_index_from_refined_core
                    outer = side_rows.iloc[pair_index + 1]
                    outer_offset = outer.offset_index_from_refined_core
                    if inner_offset not in pivot or outer_offset not in pivot:
                        continue
                    valid_pair = pivot[[inner_offset, outer_offset]].dropna()
                    if valid_pair.empty:
                        continue
                    declines = valid_pair[inner_offset] > valid_pair[outer_offset]
                    row_index = int(outer.name)
                    output.loc[row_index, "n_sections_with_outward_decline"] = int(
                        declines.sum()
                    )
                    output.loc[row_index, "fraction_sections_with_outward_decline"] = (
                        float(declines.mean())
                    )
            _, _, distances = geometry.inverse(
                np.full(len(neighbors), focal.ridge_x),
                np.full(len(neighbors), focal.ridge_y),
                neighbors.ridge_x.to_numpy(float),
                neighbors.ridge_y.to_numpy(float),
            )
            span_by_section[focal.section_id] = float(np.max(distances))
    summaries = summaries.copy()
    summaries["composite_along_branch_span_length"] = summaries.section_id.map(
        span_by_section
    )
    summaries["n_neighbor_sections"] = (
        summaries.section_id.map(
            output.groupby("section_id").n_composite_sections.max()
        )
        .fillna(0)
        .astype(int)
    )
    return output, summaries


def _first_distance_descriptor(side_rows: pd.DataFrame, threshold: float) -> float:
    selected = side_rows.loc[side_rows.U_parallel_relative.le(threshold)]
    return float(selected.outward_distance_length.iloc[0]) if not selected.empty else np.nan


def _detect_drop_zones(
    cross_sections: pd.DataFrame,
    summaries: pd.DataFrame,
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zone_records: list[dict[str, Any]] = []
    flank_records: list[dict[str, Any]] = []
    summary_updates: list[dict[str, Any]] = []
    for summary in track(
        summaries.itertuples(index=False), desc="Flux fronts: detection",
        total=len(summaries), unit="sections",
    ):
        profile = cross_sections.loc[
            cross_sections.section_id.eq(summary.section_id)
        ].copy()
        side_results: dict[str, Any] = {}
        for side in ("left", "right"):
            observable_field = f"{side}_flank_observable"
            side_observable = bool(getattr(summary, observable_field, True))
            if not side_observable:
                side_results[f"{side}_drop_detected"] = False
                side_results[f"{side}_candidate_count"] = 0
                side_results[f"{side}_drop_distance_length"] = np.nan
                side_results[f"{side}_drop_magnitude"] = np.nan
                side_results[f"{side}_along_branch_persistence"] = False
                side_results[f"{side}_distance_50pct_length"] = np.nan
                side_results[f"{side}_distance_1e_length"] = np.nan
                side_results[f"{side}_zero_crossing_length"] = np.nan
                side_results[f"{side}_flank_status"] = "flank_not_observable"
                continue
            selected = profile.loc[profile.side.isin(("core", side))].copy()
            selected["outward_distance_length"] = selected.d_from_refined_core_length.abs()
            selected = selected.sort_values("outward_distance_length")
            valid = selected.loc[selected.sample_valid].copy()
            zones: list[dict[str, Any]] = []
            for index in range(len(valid) - 1):
                inner = valid.iloc[index]
                outer = valid.iloc[index + 1]
                width = float(outer.outward_distance_length - inner.outward_distance_length)
                if width <= 0:
                    continue
                profile_field = (
                    "U_parallel_composite_median"
                    if min(inner.n_composite_sections, outer.n_composite_sections) >= 2
                    else "U_parallel_smoothed"
                )
                inner_value = float(inner[profile_field])
                outer_value = float(outer[profile_field])
                if not np.isfinite(inner_value) or not np.isfinite(outer_value):
                    continue
                absolute_drop = inner_value - outer_value
                following = valid.iloc[index + 2 : index + 4][profile_field].dropna()
                remains_lower = bool(
                    len(following)
                    and following.median() <= (inner_value + outer_value) / 2.0
                )
                recovery = (
                    max(0.0, float(following.max() - outer_value))
                    if len(following)
                    else np.nan
                )
                relative_drop = (
                    absolute_drop / summary.U_parallel_core
                    if summary.U_parallel_core > 0
                    else np.nan
                )
                slope = -absolute_drop / width
                along_persistence = bool(
                    outer.n_sections_with_outward_decline
                    >= getattr(config, "minimum_persistent_neighbor_sections", 2)
                    and outer.fraction_sections_with_outward_decline
                    >= getattr(config, "minimum_persistent_fraction", 0.5)
                )
                persistence_available = bool(outer.n_composite_sections >= 2)
                eligible = bool(
                    absolute_drop > 0
                    and remains_lower
                    and (not persistence_available or along_persistence)
                )
                zones.append(
                    {
                        "experiment_id": summary.experiment_id,
                        "support_threshold": summary.support_threshold,
                        "intensity_level": summary.intensity_level,
                        "component_id": summary.component_id,
                        "segment_id": summary.segment_id,
                        "section_id": summary.section_id,
                        "cell_id": summary.cell_id,
                        "side": side,
                        "candidate_distance_length": float(outer.outward_distance_length),
                        "candidate_x": float(outer.sample_x),
                        "candidate_y": float(outer.sample_y),
                        "U_core": summary.U_parallel_core,
                        "U_inner": inner_value,
                        "U_outer": outer_value,
                        "absolute_drop": absolute_drop,
                        "relative_drop": relative_drop,
                        "drop_slope": slope,
                        "drop_slope_rate_per_length": slope,
                        "drop_width_length": width,
                        "number_supporting_samples": int(2 + len(following)),
                        "outer_recovery": recovery,
                        "along_branch_persistence": along_persistence,
                        "n_sections_with_outward_decline": int(
                            outer.n_sections_with_outward_decline
                        )
                        if np.isfinite(outer.n_sections_with_outward_decline)
                        else 0,
                        "fraction_sections_with_outward_decline": float(
                            outer.fraction_sections_with_outward_decline
                        )
                        if np.isfinite(outer.fraction_sections_with_outward_decline)
                        else np.nan,
                        "n_neighbor_sections": int(
                            min(inner.n_composite_sections, outer.n_composite_sections)
                        ),
                        "minimum_N_out_move": float(
                            min(inner.N_out_move, outer.N_out_move)
                        ),
                        "minimum_N_in_move": float(
                            min(inner.N_in_move, outer.N_in_move)
                        ),
                        "profile_used": profile_field,
                        "candidate_eligible": eligible,
                        "nearby_branch_contamination": summary.nearby_branch_contamination,
                        "quality_flags": summary.quality_flags,
                    }
                )
            zone_frame = pd.DataFrame(zones)
            if not zone_frame.empty:
                eligible = zone_frame.loc[zone_frame.candidate_eligible].copy()
            else:
                eligible = pd.DataFrame()
            if not eligible.empty:
                eligible = eligible.sort_values(
                    [
                        "along_branch_persistence",
                        "absolute_drop",
                        "drop_slope_rate_per_length",
                    ],
                    ascending=[False, False, True],
                    kind="stable",
                )
                rank_by_index = pd.Series(
                    np.arange(1, len(eligible) + 1), index=eligible.index
                )
                zone_frame["drop_rank"] = zone_frame.index.map(rank_by_index)
                best = zone_frame.loc[zone_frame.drop_rank.eq(1)].iloc[0].to_dict()
                strong_recovery = bool(
                    np.isfinite(best["outer_recovery"])
                    and best["outer_recovery"]
                    > config.diagnostic_strong_outer_recovery_fraction
                    * best["absolute_drop"]
                )
                flank_flags = [
                    flag for flag in str(best["quality_flags"]).split(";") if flag
                ]
                _append_flag(flank_flags, len(eligible) > 1, "multiple_candidate_drops")
                _append_flag(flank_flags, strong_recovery, "strong_outer_recovery")
                _append_flag(
                    flank_flags,
                    bool((valid.U_parallel_raw.iloc[-2:] < 0).any()),
                    "opposing_outer_flux",
                )
                best["quality_flags"] = ";".join(flank_flags)
                best["multiple_candidate_drops"] = len(eligible) > 1
                best["strong_outer_recovery"] = strong_recovery
                flank_records.append(best)
                detected = True
                candidate_count = len(eligible)
                distance = best["candidate_distance_length"]
                magnitude = best["absolute_drop"]
                persistence = best["along_branch_persistence"]
            else:
                if not zone_frame.empty:
                    zone_frame["drop_rank"] = np.nan
                detected = False
                candidate_count = 0
                distance = magnitude = np.nan
                persistence = False
            if not zone_frame.empty:
                zone_records.extend(zone_frame.to_dict("records"))
            side_results[f"{side}_drop_detected"] = detected
            side_results[f"{side}_candidate_count"] = candidate_count
            side_results[f"{side}_drop_distance_length"] = distance
            side_results[f"{side}_drop_magnitude"] = magnitude
            side_results[f"{side}_along_branch_persistence"] = persistence
            side_results[f"{side}_flank_status"] = (
                "candidate_drop" if detected else "observable_no_candidate_drop"
            )
            side_results[f"{side}_distance_50pct_length"] = _first_distance_descriptor(
                valid, 0.5
            )
            side_results[f"{side}_distance_1e_length"] = _first_distance_descriptor(
                valid, 1 / np.e
            )
            side_results[f"{side}_zero_crossing_length"] = _first_distance_descriptor(
                valid, 0.0
            )
        observable_sides = [
            side
            for side in ("left", "right")
            if side_results[f"{side}_flank_status"] != "flank_not_observable"
        ]
        no_drop = bool(observable_sides) and not any(
            side_results[f"{side}_drop_detected"] for side in observable_sides
        )
        all_observable_have_drop = bool(observable_sides) and all(
            side_results[f"{side}_drop_detected"] for side in observable_sides
        )
        observable_side_without_drop = any(
            not side_results[f"{side}_drop_detected"] for side in observable_sides
        )
        multiple = (
            side_results["left_candidate_count"] > 1
            or side_results["right_candidate_count"] > 1
        )
        flags = [flag for flag in str(summary.quality_flags).split(";") if flag]
        _append_flag(flags, no_drop, "no_candidate_drop")
        _append_flag(
            flags,
            any(
                side_results[f"{side}_flank_status"] == "flank_not_observable"
                for side in ("left", "right")
            ),
            "flank_not_observable",
        )
        _append_flag(flags, multiple, "multiple_candidate_drops")
        summary_updates.append(
            {
                "section_id": summary.section_id,
                **side_results,
                "no_candidate_drop": no_drop,
                "all_observable_flanks_have_candidate_drop": all_observable_have_drop,
                "observable_flank_without_candidate_drop": observable_side_without_drop,
                "multiple_candidate_drops": multiple,
                "quality_flags_updated": ";".join(flags),
            }
        )
    zones = pd.DataFrame.from_records(zone_records)
    flanks = pd.DataFrame.from_records(flank_records)
    updates = pd.DataFrame.from_records(summary_updates)
    summaries = summaries.merge(updates, on="section_id", how="left")
    summaries["quality_flags"] = summaries.pop("quality_flags_updated")
    return zones, flanks, summaries


def compute_stage6_fields(
    stage4_cells: pd.DataFrame,
    stage5_segment_members: pd.DataFrame,
    stage5_segments: pd.DataFrame,
    grid: Any,
    *,
    stage5_config: Any,
    config: Any,
    geometry: SpatialGeometry | None = None,
    boundary_aware_branch_cores: bool = False,
    experiments: tuple[tuple[int, str], ...],
    field_variant: str = "raw",
) -> Stage6Fields:
    """Compute local profiles and independent candidate flank points."""
    required = set(SAMPLED_FIELDS) | {"theta_mu_out"}
    missing = sorted(required - set(stage4_cells.columns))
    if missing:
        raise ValueError(f"Stage 4 fields missing Stage 6 columns: {missing}")
    geometry = geometry or GeographicGeometry("WGS84", "km")
    _, _, effective_scale = _physical_cell_scales(stage4_cells, grid, geometry)
    scale_by_cell = pd.Series(effective_scale, index=stage4_cells.cell_id)
    cross_outputs: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    base_prepared_fields = {
        field_name: _grid_array(stage4_cells, grid, field_name)
        for field_name in (*SAMPLED_FIELDS, *OPTIONAL_SAMPLED_FIELDS)
        if field_name in stage4_cells
    }
    support_grids = {
        threshold: base_prepared_fields["N_out_move"] >= threshold
        for threshold in {threshold for threshold, _ in experiments}
    }
    prepared_by_support: dict[int, dict[str, np.ndarray]] = {}
    for threshold in support_grids:
        prepared = dict(base_prepared_fields)
        optional_fields = {
            "C_neigh_out": f"C_neigh_out_1_ge_{threshold}",
            "C_neigh_in": f"C_neigh_in_1_ge_{threshold}",
            "C_neigh_out_mu": f"C_neigh_out_mu_ge_{threshold}",
            "C_neigh_in_mu": f"C_neigh_in_mu_ge_{threshold}",
        }
        for alias, source in optional_fields.items():
            if source in stage4_cells:
                prepared[alias] = _grid_array(stage4_cells, grid, source)
        prepared_by_support[threshold] = prepared
    for support_threshold, intensity_level in experiments:
        experiment_id = f"s{support_threshold}_{field_variant}_{intensity_level}"
        members = stage5_segment_members.loc[
            stage5_segment_members.support_threshold.eq(support_threshold)
            & stage5_segment_members.field_variant.eq(field_variant)
            & stage5_segment_members.intensity_level.eq(intensity_level)
        ].copy()
        segment_context = stage5_segments.loc[
            stage5_segments.support_threshold.eq(support_threshold)
            & stage5_segments.field_variant.eq(field_variant)
            & stage5_segments.intensity_level.eq(intensity_level),
            ["segment_id", "maximum_local_tangent_turn_degrees"],
        ]
        members = members.merge(segment_context, on="segment_id", how="left")
        members["grid_effective_scale_length"] = members.cell_id.map(scale_by_cell)
        for _, center in track(
            members.iterrows(), desc="Flux fronts: sampling",
            total=len(members), unit="sections",
        ):
            rows, section_summary = _section_rows(
                center,
                members,
                grid,
                experiment_id=experiment_id,
                config=config,
                geometry=geometry,
                interpolation_weight_tolerance=stage5_config.interpolation_weight_tolerance,
                prepared_fields=prepared_by_support[support_threshold],
                support_grid=support_grids[support_threshold],
                boundary_aware_branch_cores=boundary_aware_branch_cores,
            )
            cross_outputs.append(rows)
            summary_records.append(section_summary)
    cross_sections = pd.concat(cross_outputs, ignore_index=True)
    section_summaries = pd.DataFrame.from_records(summary_records)
    cross_sections, section_summaries = _add_along_branch_composites(
        cross_sections,
        section_summaries,
        half_window=config.composite_half_window_sections,
        geometry=geometry,
    )
    drop_zones, flank_points, section_summaries = _detect_drop_zones(
        cross_sections, section_summaries, config
    )
    baseline_experiment_id = f"s{experiments[0][0]}_{field_variant}_{experiments[0][1]}"
    summary: dict[str, Any] = {
        "baseline_experiment": baseline_experiment_id,
        "orientation": "central theta_mu_out compass bearing; same tangent across section",
        "cross_stream_geometry": (
            f"{geometry.coordinate_system} geometry +/-5 local effective grid scales"
        ),
        "independent_sampling_interval": "one local effective grid scale",
        "core_refinement": "maximum supported U_parallel within +/-1 grid scale",
        "profile_smoother": f"contiguous rolling median window {config.robust_median_window_samples}",
        "along_branch_composite": f"median/q25/q75 within same segment +/-{config.composite_half_window_sections} sections",
        "continuous_front_lines_created": False,
        "global_gradient_comparison_implemented": False,
        "permeability_implemented": False,
        "stage7_implemented": False,
        "geographic_filters_applied": False,
        "boundary_aware_branch_cores": boundary_aware_branch_cores,
        "unobservable_flank_semantics": (
            "flank_not_observable_never_no_candidate_drop_or_zero"
            if boundary_aware_branch_cores
            else "not_applicable_all_stage5_cores_two_sided"
        ),
    }
    for experiment_id, group in section_summaries.groupby("experiment_id"):
        flanks = flank_points.loc[flank_points.experiment_id.eq(experiment_id)]
        both = group.left_drop_detected & group.right_drop_detected
        one = group.left_drop_detected ^ group.right_drop_detected
        neither = group.no_candidate_drop
        one_sided_cores = group.n_observable_flanks.eq(1)
        two_sided_cores = group.n_observable_flanks.eq(2)
        asymmetry = (group.left_drop_distance_length - group.right_drop_distance_length).abs()
        raw_variation = group.profile_total_variation_raw.replace(0, np.nan)
        variation_reduction = (
            group.profile_total_variation_raw - group.profile_total_variation_smoothed
        ) / raw_variation
        prefix = experiment_id
        summary.update(
            {
                f"{prefix}_sections": len(group),
                f"{prefix}_unique_ridge_cells": int(group.cell_id.nunique()),
                f"{prefix}_sections_with_valid_core": int(
                    group.U_parallel_core.notna().sum()
                ),
                f"{prefix}_core_unchanged": int(group.d_core_grid_scales.eq(0).sum()),
                f"{prefix}_core_shifted_one_scale": int(
                    group.d_core_grid_scales.abs().eq(1).sum()
                ),
                f"{prefix}_two_sided_drop_sections": int(both.sum()),
                f"{prefix}_one_sided_drop_sections": int(one.sum()),
                f"{prefix}_no_drop_sections": int(neither.sum()),
                f"{prefix}_two_sided_core_sections": int(two_sided_cores.sum()),
                f"{prefix}_one_sided_core_sections": int(one_sided_cores.sum()),
                f"{prefix}_unobservable_flanks": int(group.n_unobservable_flanks.sum()),
                f"{prefix}_observable_flanks": int(group.n_observable_flanks.sum()),
                f"{prefix}_one_sided_cores_with_observed_flank_drop": int(
                    (
                        one_sided_cores
                        & (group.left_drop_detected | group.right_drop_detected)
                    ).sum()
                ),
                f"{prefix}_one_sided_cores_with_observable_flank_no_drop": int(
                    (one_sided_cores & group.no_candidate_drop).sum()
                ),
                f"{prefix}_nearby_branch_contamination_sections": int(
                    group.nearby_branch_contamination.sum()
                ),
                f"{prefix}_short_sections": int(
                    group.short_available_cross_section.sum()
                ),
                f"{prefix}_curved_sections": int(
                    group.high_local_curvature_turning.sum()
                ),
                f"{prefix}_low_R1_sections": int(group.low_R1_out.sum()),
                f"{prefix}_median_left_drop_distance_length": float(
                    group.left_drop_distance_length.dropna().median()
                ),
                f"{prefix}_median_right_drop_distance_length": float(
                    group.right_drop_distance_length.dropna().median()
                ),
                f"{prefix}_median_absolute_left_right_distance_asymmetry_length": float(
                    asymmetry.dropna().median()
                ),
                f"{prefix}_median_flank_absolute_drop": float(
                    flanks.absolute_drop.dropna().median()
                ),
                f"{prefix}_along_persistent_flank_fraction": float(
                    flanks.along_branch_persistence.mean()
                ),
                f"{prefix}_flanks_with_nearby_branch_contamination": int(
                    flanks.nearby_branch_contamination.sum()
                ),
                f"{prefix}_median_raw_total_variation": float(
                    group.profile_total_variation_raw.dropna().median()
                ),
                f"{prefix}_median_smoothed_total_variation": float(
                    group.profile_total_variation_smoothed.dropna().median()
                ),
                f"{prefix}_median_fractional_variation_reduction": float(
                    variation_reduction.dropna().median()
                ),
                f"{prefix}_curved_sections_with_valid_core": int(
                    (
                        group.high_local_curvature_turning
                        & group.U_parallel_core.notna()
                    ).sum()
                ),
                f"{prefix}_no_drop_with_support_limitation": int(
                    (
                        neither
                        & (
                            group.n_missing_support_samples.gt(0)
                            | group.short_available_cross_section
                        )
                    ).sum()
                ),
                f"{prefix}_no_drop_with_nearby_branch": int(
                    (neither & group.nearby_branch_contamination).sum()
                ),
            }
        )
    return Stage6Fields(
        cross_sections=cross_sections,
        section_summaries=section_summaries,
        candidate_drop_zones=drop_zones,
        candidate_flank_points=flank_points,
        summary=summary,
    )
