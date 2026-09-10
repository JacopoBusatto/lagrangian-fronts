from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from lagrangian_fronts import load_config, read_trajectories
from lagrangian_fronts.config import TimeConfig, TrajectoryIdConfig
from lagrangian_fronts.trajectories import resolve_input_paths


@pytest.mark.parametrize("fmt", ["csv", "parquet", "netcdf", "zarr"])
@pytest.mark.parametrize("kind", ["datetime", "numeric"])
def test_formats_custom_fields_and_time(config, tmp_path, fmt, kind):
    suffix = {"netcdf": "nc"}.get(fmt, fmt)
    path = tmp_path / f"tracks.{suffix}"
    times = (
        pd.date_range("2026-01-01", periods=3, freq="2s")
        if kind == "datetime"
        else np.array([0.0, 2.0, 4.0])
    )
    frame = pd.DataFrame(
        {
            "track": ["001"] * 3,
            "t": times,
            "X": [0.2, 1.2, 2.2],
            "Y": [0.2] * 3,
            "unused": [99] * 3,
        }
    )
    if fmt == "csv":
        frame.to_csv(path, index=False)
    elif fmt == "parquet":
        frame.to_parquet(path, index=False)
    else:
        dataset = xr.Dataset({c: ("sample", frame[c].to_numpy()) for c in frame})
        if fmt == "netcdf":
            dataset.to_netcdf(path)
        else:
            dataset.to_zarr(path)
    config = replace(
        config,
        input=replace(
            config.input,
            paths=(str(path),),
            time=TimeConfig(kind, "s" if kind == "numeric" else None),
        ),
    )
    result = read_trajectories(config)
    assert set(result.table) == {"trajectory", "time", "x", "y"}
    assert result.table.time.tolist() == [0.0, 2.0, 4.0]
    assert result.table.x.tolist() == [0.2, 1.2, 2.2]
    assert result.diagnostics["source_trajectories"] == 1
    assert "001" in result.table.trajectory.iloc[0]


@pytest.mark.parametrize("fmt", ["netcdf", "zarr"])
def test_rectangular_dimension_and_coordinate_fields(config, tmp_path, fmt):
    path = tmp_path / ("rect.nc" if fmt == "netcdf" else "rect.zarr")
    ds = xr.Dataset(
        {
            "X": (("track", "obs"), [[0.2, 1.2], [2.2, 3.2]]),
            "Y": (("track", "obs"), [[0.2, 0.2], [0.2, 0.2]]),
            "t": ("obs", [0.0, 2.0]),
        }
    )
    if fmt == "netcdf":
        ds.to_netcdf(path)
    else:
        ds.to_zarr(path)
    config = replace(
        config,
        input=replace(
            config.input,
            paths=(str(path),),
            columns=replace(config.input.columns, obs="obs"),
        ),
    )
    result = read_trajectories(config)
    assert len(result.table) == 4
    assert result.diagnostics["source_trajectories"] == 2
    assert result.table.obs.tolist() == [0, 1, 0, 1]


def test_unrelated_xarray_dimensions_are_rejected(config, tmp_path):
    path = tmp_path / "bad.nc"
    xr.Dataset(
        {
            "X": ("a", [0.2, 1.2]),
            "Y": ("b", [0.2, 1.2]),
            "t": ("a", [0.0, 2.0]),
            "track": ("a", [1, 1]),
        }
    ).to_netcdf(path)
    config = replace(config, input=replace(config.input, paths=(str(path),)))
    with pytest.raises(ValueError, match="incompatible observation dimensions"):
        read_trajectories(config)


def test_globs_scope_and_deduplication(config, tmp_path):
    for name, times in (("b.csv", [2, 3]), ("a.csv", [0, 1])):
        pd.DataFrame(
            {"track": [1, 1], "t": times, "X": [0.2, 0.3], "Y": [0.2, 0.2]}
        ).to_csv(tmp_path / name, index=False)
    config = replace(config, input=replace(config.input, paths=("*.csv", "a.csv")))
    assert [p.name for p in resolve_input_paths(config)] == ["a.csv", "b.csv"]
    separate = read_trajectories(config)
    assert separate.diagnostics["source_trajectories"] == 2
    assert separate.table.time.tolist() == [0.0, 1.0, 0.0, 1.0]
    global_config = replace(
        config,
        input=replace(config.input, trajectory_id=TrajectoryIdConfig(scope="global")),
    )
    merged = read_trajectories(global_config)
    assert merged.diagnostics["source_trajectories"] == 1
    assert merged.table.time.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_filename_identity_distinguishes_same_basename(config, tmp_path):
    paths = []
    for directory in ("a", "b"):
        folder = tmp_path / directory
        folder.mkdir()
        path = folder / "track.csv"
        pd.DataFrame({"t": [0, 2], "X": [0.2, 0.3], "Y": [0.2, 0.2]}).to_csv(
            path, index=False
        )
        paths.append(str(path))
    config = replace(
        config,
        input=replace(
            config.input,
            paths=tuple(paths),
            trajectory_id=TrajectoryIdConfig(mode="filename"),
            columns=replace(config.input.columns, trajectory=None),
        ),
    )
    first, second = read_trajectories(config), read_trajectories(config)
    pd.testing.assert_frame_equal(first.table, second.table)
    assert first.table.trajectory.nunique() == 2


def test_sanitation_duplicates_obs_and_short_tracks(config, tmp_path):
    frame = pd.DataFrame(
        {
            "track": [1, 1, 1, 1, 2, 3, 4],
            "t": [2, 0, 0, 4, 0, 0, 0],
            "X": [1.2, 0.8, 0.2, np.inf, 0.2, 0.2, 0.2],
            "Y": [0.2] * 7,
            "obs": [0, 2, 1, 0, 0, 0, 0],
        }
    )
    frame.loc[5, "t"] = np.nan
    frame.loc[6, "track"] = np.nan
    frame.to_csv(tmp_path / "tracks.csv", index=False)
    config = replace(
        config,
        input=replace(config.input, columns=replace(config.input.columns, obs="obs")),
    )
    result = read_trajectories(config)
    assert result.table.x.tolist() == [0.2, 1.2]
    assert result.diagnostics["invalid_observations"] == 3
    assert result.diagnostics["duplicate_time_observations"] == 1
    assert result.diagnostics["discarded_short_trajectories"] == 1
    assert result.diagnostics["discarded_empty_trajectories"] == 1


def test_group_members_do_not_merge(config, tmp_path):
    pd.DataFrame(
        {
            "track": [1] * 4,
            "t": [0, 2, 0, 2],
            "X": [0.2, 0.3, 1.2, 1.3],
            "Y": [0.2] * 4,
            "member": [0, 0, 1, 1],
        }
    ).to_csv(tmp_path / "tracks.csv", index=False)
    config = replace(
        config,
        input=replace(
            config.input, columns=replace(config.input.columns, group_member="member")
        ),
    )
    result = read_trajectories(config)
    assert result.diagnostics["retained_trajectories"] == 2
    assert len(result.table) == 4


@pytest.mark.parametrize(
    "unit,seconds", [("s", 1), ("min", 60), ("h", 3600), ("day", 86400)]
)
def test_numeric_units(config, tmp_path, unit, seconds):
    pd.DataFrame(
        {"track": [1, 1], "t": [100, 101], "X": [0.2, 0.3], "Y": [0.2, 0.2]}
    ).to_csv(tmp_path / "tracks.csv", index=False)
    config = replace(
        config, input=replace(config.input, time=TimeConfig("numeric", unit))
    )
    assert read_trajectories(config).table.time.tolist() == [0.0, float(seconds)]


def test_missing_input_and_fields(config, tmp_path):
    with pytest.raises(FileNotFoundError, match="No trajectory input matches"):
        read_trajectories(config)
    pd.DataFrame({"t": [0, 2], "X": [0.2, 0.3], "Y": [0.2, 0.2]}).to_csv(
        tmp_path / "tracks.csv", index=False
    )
    with pytest.raises(ValueError, match="Cannot read trajectories"):
        read_trajectories(config)


def test_yaml_glob_paths_resolve_from_config_directory(config, tmp_path):
    pd.DataFrame(
        {"track": [1, 1], "t": [0, 2], "X": [0.2, 0.3], "Y": [0.2, 0.2]}
    ).to_csv(tmp_path / "tracks.csv", index=False)
    raw = config.to_dict()
    raw["input"]["paths"] = ["tracks*.csv", "tracks.csv"]
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(raw))
    resolved = load_config(path)
    assert read_trajectories(resolved).input_files == (
        str((tmp_path / "tracks.csv").resolve()),
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: raw.pop("config_version"), "config_version"),
        (lambda raw: raw.update(config_version=2), "config_version"),
        (lambda raw: raw.update(config_version=True), "config_version"),
        (lambda raw: raw.update(transport=raw.pop("flux")), "replace transport: with flux:"),
        (lambda raw: raw.update(transport=None), "replace transport: with flux:"),
        (lambda raw: raw["matrix"].update(compute="false"), "invalid value type"),
        (lambda raw: raw["matrix"].update(timestep=0), "positive"),
        (lambda raw: raw["matrix"].update(timestep=float("nan")), "finite"),
        (lambda raw: raw["matrix"].update(max_interpolation_gap=-1), "positive"),
        (lambda raw: raw["input"].update(unknown=1), "unknown input keys"),
        (lambda raw: raw["input"]["time"].update(unit="week"), "input.time.unit"),
        (lambda raw: raw["grid"].update(x_max=4.5), "integer multiple"),
    ],
)
def test_versioned_strict_config(config, tmp_path, mutate, match):
    raw = config.to_dict()
    mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError), match=match):
        load_config(path)
