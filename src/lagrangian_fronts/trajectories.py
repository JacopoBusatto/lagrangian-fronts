"""Format adapters and structural sanitation for scientifically prepared tracks."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import AnalysisConfig, InputConfig

SECONDS_PER_UNIT = {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0}


@dataclass(frozen=True)
class TrajectoryData:
    table: pd.DataFrame
    diagnostics: dict
    input_files: tuple[str, ...]


def resolve_input_paths(config: AnalysisConfig) -> tuple[Path, ...]:
    if config.input is None:
        raise ValueError("trajectory input is required")
    base = Path(config.input_base or Path.cwd())
    paths = {}
    for pattern in config.input.paths:
        candidate = Path(pattern).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        matches = glob.glob(str(candidate), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No trajectory input matches: {candidate}")
        for match in matches:
            path = Path(match).resolve()
            paths[os.path.normcase(str(path))] = path
    return tuple(paths[k] for k in sorted(paths))


def _mappings(config: InputConfig) -> dict[str, str]:
    mapping = {
        key: value for key, value in asdict(config.columns).items() if value is not None
    }
    if config.trajectory_id.mode == "filename":
        mapping.pop("trajectory", None)
    return mapping


def _read_csv(path: Path, config: InputConfig) -> pd.DataFrame:
    mapping = _mappings(config)
    ids = {
        v: "string" for k, v in mapping.items() if k in {"trajectory", "group_member"}
    }
    return pd.read_csv(path, usecols=list(mapping.values()), dtype=ids)


def _read_parquet(path: Path, config: InputConfig) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(_mappings(config).values()))


def _read_xarray(path: Path, config: InputConfig) -> pd.DataFrame:
    opener = xr.open_zarr if path.is_dir() else xr.open_dataset
    with opener(
        path, chunks=None, decode_times=config.time.kind == "datetime"
    ) as dataset:
        mapping = _mappings(config)
        missing = set(mapping.values()) - set(dataset.variables) - set(dataset.dims)
        if missing:
            raise ValueError(f"missing mapped fields: {sorted(missing)}")
        arrays = {key: dataset[name] for key, name in mapping.items()}
        xd, yd = arrays["x"].dims, arrays["y"].dims
        if not (set(xd) <= set(yd) or set(yd) <= set(xd)):
            raise ValueError(
                "x/y have incompatible observation dimensions; prepare a trajectory table upstream"
            )
        dims = xd if len(xd) >= len(yd) else yd
        if not dims or any(not set(a.dims) <= set(dims) for a in arrays.values()):
            raise ValueError(
                "mapped fields introduce incompatible observation dimensions"
            )
        template = arrays["x"] if len(xd) >= len(yd) else arrays["y"]
        columns = {}
        for key, array in arrays.items():
            if key == "time" and config.time.kind == "datetime":
                calendar = array.encoding.get(
                    "calendar", array.attrs.get("calendar", "standard")
                )
                if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
                    raise ValueError(
                        f"unsupported datetime calendar: {calendar}; convert to numeric time upstream"
                    )
            columns[mapping[key]] = (
                array.broadcast_like(template).transpose(*dims).values.reshape(-1)
            )
        return pd.DataFrame(columns)


READERS: dict[str, Callable[[Path, InputConfig], pd.DataFrame]] = {
    "csv": _read_csv,
    "parquet": _read_parquet,
    "netcdf": _read_xarray,
    "zarr": _read_xarray,
}
EXTENSIONS = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".netcdf": "netcdf",
    ".zarr": "zarr",
}


def _identity(values: pd.Series) -> pd.Series:
    # Scalar ID values become strings; missing IDs stay missing, not the string 'nan'.
    return values.astype("string").replace(r"^\s*$", pd.NA, regex=True)


def _file_identity(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # Windows inputs and YAML on different drives.
        return path.as_posix()


def read_trajectories(config: AnalysisConfig) -> TrajectoryData:
    """Read selected fields once; canonical time is elapsed seconds per track."""
    paths = resolve_input_paths(config)
    inp = config.input
    mapping = _mappings(inp)
    frames = []
    for path in paths:
        fmt = (
            inp.format if inp.format != "auto" else EXTENSIONS.get(path.suffix.lower())
        )
        if fmt not in READERS:
            raise ValueError(
                f"Cannot detect trajectory format for {path}; set input.format"
            )
        if path.is_dir() != (fmt == "zarr"):
            raise ValueError(
                f"{fmt} requires {'a directory store' if fmt == 'zarr' else 'a file'}: {path}"
            )
        try:
            frame = READERS[fmt](path, inp).rename(
                columns={v: k for k, v in mapping.items()}
            )
        except (ValueError, KeyError, OSError) as exc:
            raise ValueError(f"Cannot read trajectories from {path}: {exc}") from exc
        token = _file_identity(path, Path(config.input_base or Path.cwd()))
        if inp.trajectory_id.mode == "filename":
            frame["trajectory"] = json.dumps([token], ensure_ascii=False)
        else:
            ids = _identity(frame["trajectory"])
            if inp.trajectory_id.scope == "file":
                # Construct each encoded identity once per file, not once per observation.
                lookup = {
                    v: json.dumps([token, v], ensure_ascii=False)
                    for v in ids.dropna().unique()
                }
                ids = ids.map(lookup)
            frame["trajectory"] = ids
        if "group_member" in frame:
            frame["group_member"] = _identity(frame["group_member"])
        frames.append(frame)
    work = pd.concat(frames, ignore_index=True)
    del frames
    groups = ["trajectory"] + (["group_member"] if "group_member" in work else [])
    identities_valid = work[groups].notna().all(axis=1)
    source_tracks = len(work.loc[identities_valid, groups].drop_duplicates())
    for key in ("x", "y"):
        work[key] = pd.to_numeric(work[key], errors="coerce").astype(float)
    if inp.time.kind == "datetime":
        # Do not reinterpret bare numeric inputs as nanoseconds since the Unix epoch.
        if pd.api.types.is_numeric_dtype(work.time):
            raise ValueError(
                "datetime input contains numeric time; configure numeric time or CF datetime decoding"
            )
        work["time"] = pd.to_datetime(
            work.time, errors="coerce", utc=True, format="mixed"
        )
        valid_time = work.time.notna()
    else:
        if pd.api.types.is_datetime64_any_dtype(work.time):
            raise ValueError("numeric input contains datetime values")
        work["time"] = pd.to_numeric(work.time, errors="coerce")
        valid_time = np.isfinite(work.time)
    valid = identities_valid & valid_time & np.isfinite(work.x) & np.isfinite(work.y)
    if config.geometry.coordinate_system == "geographic":
        valid &= work.y.between(-90, 90)
    invalid_count = int((~valid).sum())
    raw_count = len(work)
    work = work.loc[valid].copy()
    sort_keys = groups + ["time"] + (["obs"] if "obs" in work else [])
    try:
        work = work.sort_values(sort_keys, kind="stable", na_position="last")
    except TypeError as exc:
        raise ValueError(
            "observation sort keys must have consistently comparable values"
        ) from exc
    duplicate = work.duplicated(groups + ["time"], keep="first")
    duplicate_count = int(duplicate.sum())
    work = work.loc[~duplicate].copy()
    sizes = work.groupby(groups, observed=True, sort=False).time.transform("size")
    short_tracks = len(work.loc[sizes < 2, groups].drop_duplicates())
    short_observations = int((sizes < 2).sum())
    work = work.loc[sizes >= 2].copy()
    origin = work.groupby(groups, observed=True, sort=False).time.transform("min")
    if inp.time.kind == "datetime":
        work["time"] = (work.time - origin).dt.total_seconds()
    else:
        work["time"] = (work.time - origin).astype(float) * SECONDS_PER_UNIT[
            inp.time.unit
        ]
    if not np.isfinite(work.time).all():
        raise ValueError("elapsed time overflows the numeric representation")
    work["trajectory"] = work.trajectory.astype("category")
    work = work.reset_index(drop=True)
    retained_tracks = len(work[groups].drop_duplicates())
    diagnostics = {
        "number_of_resolved_input_files": len(paths),
        "source_trajectories": source_tracks,
        "input_observations": raw_count,
        "valid_observations": len(work),
        "structurally_rejected_observations": invalid_count + duplicate_count,
        "invalid_observations": invalid_count,
        "duplicate_time_observations": duplicate_count,
        "discarded_short_trajectories": short_tracks,
        "discarded_empty_trajectories": source_tracks - short_tracks - retained_tracks,
        "observations_in_short_trajectories": short_observations,
        "retained_trajectories": retained_tracks,
        "canonical_time_unit": "s",
        "duplicate_time_policy": "first_after_stable_time_obs_sort",
    }
    return TrajectoryData(work, diagnostics, tuple(str(p) for p in paths))
