import ast
import json
import logging
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from lagrangian_fronts import load_config, read_trajectories, run
from lagrangian_fronts.io import DIRECTIONAL_TABLES, STANDARD_TABLES
from lagrangian_fronts.matrix import read_transition_matrix

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/lagrangian_fronts"


@pytest.mark.parametrize("mode", ["geographic", "cartesian"])
def test_cli_compute_and_load_are_scientifically_identical(tmp_path, mode):
    config = load_config(ROOT / f"examples/{mode}/config.yaml")
    config = replace(
        config,
        output=replace(config.output, root=str(tmp_path), run_name="computed"),
        plotting=replace(config.plotting, enabled=False),
    )
    raw = config.to_dict()
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    original = load_config(path)
    process = subprocess.run(
        [sys.executable, "-m", "lagrangian_fronts", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    computed = tmp_path / "computed"
    assert process.stdout.strip() == str(computed)
    assert "Flux fronts completed in " in process.stderr
    matrix_path = computed / "matrix/transition_matrix.parquet"
    raw["matrix"].update(compute=False, path=str(matrix_path))
    raw.pop("input")
    raw["output"]["run_name"] = "loaded"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    process = subprocess.run(
        [sys.executable, "-m", "lagrangian_fronts", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    loaded = tmp_path / "loaded"
    assert process.stdout.strip() == str(loaded)
    assert "Run completed in " in process.stderr
    assert "sections/s" not in process.stderr
    for name in (*STANDARD_TABLES, *DIRECTIONAL_TABLES, "gradient_validation.parquet"):
        actual = pd.read_parquet(loaded / "analysis" / name)
        expected = pd.read_parquet(computed / "analysis" / name)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for name in ("flux_fronts.parquet", "directional_fronts.parquet"):
        assert pd.read_parquet(computed / "analysis" / name).front_detected.any()
    assert not (loaded / "matrix/transition_matrix.parquet").exists()
    manifest = json.loads((computed / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["input"]["resolved_files"]
    assert manifest["software"]["scientific_baseline"].startswith("10e63b4")
    assert manifest["counts"]["probable_flux_fronts"] > 0
    assert "flux_directional_comparison" in manifest
    assert f"flux_threshold_{config.rate_suffix}" in manifest["selection"]
    assert "transport" not in (computed / "resolved_config.yaml").read_text()
    for name in (*STANDARD_TABLES, *DIRECTIONAL_TABLES, "gradient_validation.parquet"):
        table = pd.read_parquet(computed / "analysis" / name)
        assert all("transport" not in column for column in table.columns)
    assert set(manifest["output_inventory"]) == {
        p.relative_to(computed).as_posix() for p in computed.rglob("*") if p.is_file()
    }
    resolved = load_config(computed / "resolved_config.yaml")
    assert resolved.flux == original.flux
    pd.testing.assert_frame_equal(
        read_trajectories(resolved).table, read_trajectories(original).table
    )
    with pytest.raises(FileExistsError):
        run(load_config(path))
    config = load_config(path)
    with pytest.raises(ValueError, match="elapsed_seconds"):
        read_transition_matrix(
            matrix_path, replace(config, matrix=replace(config.matrix, timestep=999.0))
        )
    sidecar = matrix_path.with_name("matrix_summary.yaml")
    summary = yaml.safe_load(sidecar.read_text())
    summary["matrix_sha256"] = "wrong"
    sidecar.write_text(yaml.safe_dump(summary))
    with pytest.raises(ValueError, match="fingerprint"):
        read_transition_matrix(matrix_path, config)


def test_compute_uses_in_memory_matrix(config, tmp_path, monkeypatch, capsys):
    pd.DataFrame(
        {"track": [1, 1], "t": [0, 2], "X": [0.2, 1.2], "Y": [0.2, 0.2]}
    ).to_csv(tmp_path / "tracks.csv", index=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("computed matrix must not be reread")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    destination = run(config)
    assert (destination / "matrix/transition_matrix.parquet").is_file()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_loop_progress_counts_rendering_and_cleanup(size, failure, monkeypatch, caplog):
    from lagrangian_fronts import _progress

    stream, bars, clock = StringIO(), [], [0.0]
    monkeypatch.setattr(stream, "isatty", lambda: True)
    monkeypatch.setattr(sys, "stderr", stream)
    real_tqdm = _progress.tqdm

    def make_bar(**kwargs):
        bar = real_tqdm(**kwargs)
        bar._time = lambda: clock[0]
        bar.start_t = bar.last_print_t = 0.0
        bars.append(bar)
        return bar

    monkeypatch.setattr(_progress, "tqdm", make_bar)
    expected_error = pytest.raises(failure) if size and failure else nullcontext()
    with caplog.at_level(logging.INFO, logger="lagrangian_fronts"), expected_error:
        for item in _progress.track(range(size), desc="Sampling", total=size, unit="sections"):
            assert bars[0].n == item  # The current item has not completed yet.
            if item == 1:
                assert "1/3" in stream.getvalue()
                if failure:
                    raise failure("interrupted work")
            clock[0] += 0.25  # Advance the display deterministically without sleeps.
    if size:
        assert bars[0].n == (1 if failure else size)
        assert bars[0].disable  # tqdm closes and clears the bar, including on errors.
        assert "\r" in stream.getvalue()
    else:
        assert bars == [] and stream.getvalue() == ""


@pytest.mark.parametrize("terminal,level", [(False, logging.INFO), (True, logging.WARNING)])
def test_loop_progress_is_quiet_outside_cli_terminal(terminal, level, monkeypatch, caplog):
    from lagrangian_fronts._progress import track

    stream = StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: terminal)
    monkeypatch.setattr(sys, "stderr", stream)
    with caplog.at_level(level, logger="lagrangian_fronts"):
        assert list(track(range(3), desc="Sampling", total=3, unit="sections")) == [0, 1, 2]
    assert stream.getvalue() == ""


def test_empty_computed_matrix_records_failure(config, tmp_path):
    pd.DataFrame(
        {"track": [1, 1], "t": [0, 2], "X": [0.2, 10.0], "Y": [0.2, 0.2]}
    ).to_csv(tmp_path / "tracks.csv", index=False)
    with pytest.raises(ValueError, match="empty_transition_table"):
        run(config)
    destination = tmp_path / "outputs/test"
    assert json.loads((destination / "manifest.json").read_text())["status"] == "failed"
    assert (
        yaml.safe_load((destination / "matrix/matrix_summary.yaml").read_text())[
            "retained_in_domain_transitions"
        ]
        == 0
    )


def test_stay_only_matrix_preserves_baseline_analysis_requirement(config, tmp_path):
    pd.DataFrame(
        {"track": [1, 1], "t": [0, 2], "X": [0.2, 0.3], "Y": [0.2, 0.2]}
    ).to_csv(tmp_path / "tracks.csv", index=False)
    with pytest.raises(ValueError, match="requires at least one moving transition"):
        run(replace(config, run_validation=True))
    destination = tmp_path / "outputs/test"
    matrix = pd.read_parquet(destination / "matrix/transition_matrix.parquet")
    assert matrix.transition_probability.tolist() == [1.0]
    assert json.loads((destination / "manifest.json").read_text())["status"] == "failed"


def test_module_has_no_repository_imports_and_cli_has_no_overrides():
    for path in MODULE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name.startswith(("kinematicparcels", "research.")) for name in names
            ), path
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lagrangian_fronts",
            "--config",
            "unused.yaml",
            "--timestep",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
