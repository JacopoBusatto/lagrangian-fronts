"""Sparse fixed-lag transitions, independent of file schemas and pandas dates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ._statistics_kernel import validate_transition_table
from .config import AnalysisConfig
from .statistics import normalize_transition_table
from .trajectories import SECONDS_PER_UNIT, TrajectoryData


@dataclass(frozen=True)
class MatrixResult:
    transition_table: pd.DataFrame
    diagnostics: dict


def _points(times, x, y, targets, *, geographic, max_gap):
    """Vectorized endpoint lookup; statuses: exact, interpolated, missing, gap."""
    right = np.searchsorted(times, targets, side="left")
    # Snap only floating-point arithmetic roundoff, never a scientific time window.
    nearby = np.clip(right, 0, len(times) - 1)
    previous = np.maximum(nearby - 1, 0)
    use_previous = np.abs(times[previous] - targets) < np.abs(times[nearby] - targets)
    nearest = np.where(use_previous, previous, nearby)
    tolerance = 8 * np.spacing(np.maximum(np.abs(targets), np.abs(times[nearest])))
    exact = np.abs(times[nearest] - targets) <= tolerance
    status = np.full(len(targets), "missing", dtype="U12")
    out_x, out_y = np.full(len(targets), np.nan), np.full(len(targets), np.nan)
    out_x[exact], out_y[exact] = x[nearest[exact]], y[nearest[exact]]
    status[exact] = "exact"
    bracketed = ~exact & (right > 0) & (right < len(times))
    indexes = np.flatnonzero(bracketed)
    r = right[indexes]
    left = r - 1
    gap = times[r] - times[left]
    allowed = np.ones(len(indexes), dtype=bool) if max_gap is None else gap <= max_gap
    status[indexes[~allowed]] = "gap"
    indexes, r, left, gap = indexes[allowed], r[allowed], left[allowed], gap[allowed]
    fraction = (targets[indexes] - times[left]) / gap
    delta = x[r] - x[left]
    if geographic:
        # np.unwrap's shortest longitude arc, including its +/-180 tie convention.
        pair = np.unwrap(np.deg2rad(np.column_stack((x[left], x[r]))), axis=1)
        delta = np.rad2deg(pair[:, 1] - pair[:, 0])
    out_x[indexes] = x[left] + fraction * delta
    out_y[indexes] = y[left] + fraction * (y[r] - y[left])
    status[indexes] = "interpolated"
    return out_x, out_y, status


def _bins(x, y, config):
    grid = config.grid
    x = np.asarray(x, dtype=float).copy()
    if config.geometry.coordinate_system == "geographic":
        # Equivalent longitudes are expressed on the branch beginning at lon_min.
        x = (x - grid.x_min) % 360.0 + grid.x_min
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= grid.x_min)
        & (x < grid.x_max)
        & (y >= grid.y_min)
        & (y < grid.y_max)
    )
    ix, iy = np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64)
    ix[valid] = np.floor((x[valid] - grid.x_min) / grid.dx).astype(np.int64)
    iy[valid] = np.floor((y[valid] - grid.y_min) / grid.dy).astype(np.int64)
    valid &= (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    return ix, iy, valid


def validate_matrix(table, config):
    result = validate_transition_table(
        normalize_transition_table(table, config.geometry.coordinate_system),
        config.grid,
        config.validation,
    )
    if result.errors:
        raise ValueError(f"Transition-matrix validation failed: {result.errors}")
    return result.summary


def matrix_metadata(config):
    return {
        "coordinate_system": config.geometry.coordinate_system,
        "geometry": asdict(config.geometry),
        "grid": asdict(config.grid),
        "timestep": config.matrix.timestep,
        "time_unit": config.matrix.time_unit,
        "elapsed_seconds": config.matrix.timestep
        * SECONDS_PER_UNIT[config.matrix.time_unit],
        "segment_mode": config.matrix.segment_mode,
        "max_interpolation_gap": config.matrix.max_interpolation_gap,
    }


def compute_transition_matrix(
    trajectories: TrajectoryData | pd.DataFrame, config: AnalysisConfig
) -> MatrixResult:
    table = (
        trajectories.table if isinstance(trajectories, TrajectoryData) else trajectories
    )
    required = {"trajectory", "time", "x", "y"}
    if required - set(table):
        raise ValueError(
            f"missing canonical trajectory fields: {sorted(required - set(table))}"
        )
    if (
        not pd.api.types.is_numeric_dtype(table.time)
        or not np.isfinite(table[["time", "x", "y"]].to_numpy(float)).all()
    ):
        raise ValueError(
            "canonical time/x/y must be finite numeric values; time is elapsed seconds"
        )
    groups = ["trajectory"] + (["group_member"] if "group_member" in table else [])
    if table[groups].isna().any().any():
        raise ValueError("canonical trajectory identities must not be missing")
    lag = config.matrix.timestep * SECONDS_PER_UNIT[config.matrix.time_unit]
    max_gap = config.matrix.max_interpolation_gap
    if max_gap is not None:
        max_gap *= SECONDS_PER_UNIT[config.matrix.time_unit]
    counts, source_gaps = Counter(), Counter()
    diagnostics = {
        "candidate_transitions": 0,
        "exact_endpoints": 0,
        "interpolated_endpoints": 0,
        "missing_endpoints": 0,
        "gap_rejected_endpoints": 0,
        "rejected_missing_endpoint": 0,
        "rejected_interpolation_gap": 0,
        "retained_in_domain_transitions": 0,
        "rejected_outside_domain": 0,
        "outside_start": 0,
        "outside_end": 0,
    }
    for _, track in table.groupby(groups, observed=True, sort=False):
        times = track.time.to_numpy(dtype=float)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(
                "canonical tracks must contain at least two observations with strictly increasing time"
            )
        times = times - times[0]
        gaps, frequencies = np.unique(np.diff(times), return_counts=True)
        source_gaps.update(dict(zip(gaps.tolist(), frequencies.tolist())))
        x, y = track.x.to_numpy(float), track.y.to_numpy(float)
        if config.matrix.segment_mode == "non_overlapping":
            n = int(np.floor(np.nextafter(times[-1] / lag, np.inf)))
            starts = np.arange(n, dtype=float) * lag
        else:
            starts = times[:-1]
        ends = starts + lag
        diagnostics["candidate_transitions"] += len(starts)
        sx, sy, ss = _points(
            times,
            x,
            y,
            starts,
            geographic=config.geometry.coordinate_system == "geographic",
            max_gap=max_gap,
        )
        ex, ey, es = _points(
            times,
            x,
            y,
            ends,
            geographic=config.geometry.coordinate_system == "geographic",
            max_gap=max_gap,
        )
        for status, key in (
            ("exact", "exact_endpoints"),
            ("interpolated", "interpolated_endpoints"),
            ("missing", "missing_endpoints"),
            ("gap", "gap_rejected_endpoints"),
        ):
            diagnostics[key] += int(np.sum(ss == status) + np.sum(es == status))
        missing = (ss == "missing") | (es == "missing")
        gap_rejected = ~missing & ((ss == "gap") | (es == "gap"))
        diagnostics["rejected_missing_endpoint"] += int(missing.sum())
        diagnostics["rejected_interpolation_gap"] += int(gap_rejected.sum())
        available = ~missing & ~gap_rejected
        sxi, syi, inside_start = _bins(sx, sy, config)
        exi, eyi, inside_end = _bins(ex, ey, config)
        valid = available & inside_start & inside_end
        diagnostics["outside_start"] += int((available & ~inside_start).sum())
        diagnostics["outside_end"] += int((available & ~inside_end).sum())
        diagnostics["rejected_outside_domain"] += int((available & ~valid).sum())
        diagnostics["retained_in_domain_transitions"] += int(valid.sum())
        keys, frequencies = np.unique(
            np.column_stack((syi[valid], sxi[valid], eyi[valid], exi[valid])),
            axis=0,
            return_counts=True,
        )
        counts.update({tuple(key): int(n) for key, n in zip(keys, frequencies)})
    names = (
        ("lon", "lat")
        if config.geometry.coordinate_system == "geographic"
        else ("x", "y")
    )
    xx, yy = names
    keys = np.asarray(sorted(counts), dtype=np.int64).reshape(-1, 4)
    syi, sxi, eyi, exi = keys.T
    count = np.array([counts[tuple(key)] for key in keys], dtype=np.int64)
    result = pd.DataFrame(
        {
            f"start_{xx}_bin": sxi,
            f"start_{yy}_bin": syi,
            f"end_{xx}_bin": exi,
            f"end_{yy}_bin": eyi,
            f"start_{xx}_center": config.grid.x_min + (sxi + 0.5) * config.grid.dx,
            f"start_{yy}_center": config.grid.y_min + (syi + 0.5) * config.grid.dy,
            f"end_{xx}_center": config.grid.x_min + (exi + 0.5) * config.grid.dx,
            f"end_{yy}_center": config.grid.y_min + (eyi + 0.5) * config.grid.dy,
            "transition_count": count,
        }
    )
    denominator = result.groupby(
        [f"start_{xx}_bin", f"start_{yy}_bin"]
    ).transition_count.transform("sum")
    result["transition_probability"] = (result.transition_count / denominator).astype(
        float
    )
    summary = (
        validate_matrix(result, config)
        if len(result)
        else {
            "populated_start_cells": 0,
            "n_sparse_links": 0,
            "normalization_residual_max_abs": None,
        }
    )
    inferred = (
        min(source_gaps, key=lambda gap: (-source_gaps[gap], gap))
        if source_gaps
        else None
    )
    source_diagnostics = (
        trajectories.diagnostics if isinstance(trajectories, TrajectoryData) else {}
    )
    diagnostics = {
        **matrix_metadata(config),
        **source_diagnostics,
        **diagnostics,
        **summary,
        "source_timestep_seconds": inferred,
        "endpoint_counter_unit": "segment boundary evaluations (start and end; shared boundaries count twice)",
    }
    return MatrixResult(result, diagnostics)


def read_transition_matrix(path: str | Path, config: AnalysisConfig) -> pd.DataFrame:
    from .io import sha256

    path = Path(path)
    table = pd.read_parquet(path)
    validate_matrix(table, config)
    sidecar_path = path.with_name("matrix_summary.yaml")
    if sidecar_path.exists():
        summary = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("summary_version") != 1:
            raise ValueError("unsupported matrix_summary.yaml schema")
        if summary.get("matrix_sha256") != sha256(path):
            raise ValueError(
                "matrix_summary.yaml fingerprint does not match the matrix"
            )
        expected = matrix_metadata(config)
        for key in ("coordinate_system", "geometry", "grid", "elapsed_seconds"):
            if summary.get(key) != expected[key]:
                raise ValueError(
                    f"matrix_summary.yaml {key} does not match YAML configuration"
                )
    return table
