"""Acceptance checks for the installed package and both command entry points."""

import json
import os
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest
import yaml

import lagrangian_fronts


def console_script():
    suffix = ".exe" if os.name == "nt" else ""
    return str(Path(sysconfig.get_path("scripts")) / ("lagrangian-fronts" + suffix))


def test_version_and_distribution_agree():
    assert lagrangian_fronts.__version__ == "0.1.0"
    distribution = metadata.distribution("lagrangian-fronts")
    assert distribution.version == lagrangian_fronts.__version__
    assert any(entry.name == "lagrangian-fronts" and
               entry.value == "lagrangian_fronts.cli:main"
               for entry in distribution.entry_points)
    assert not any("kinematicparcels" in requirement.lower()
                   for requirement in distribution.requires or [])


@pytest.mark.parametrize("entrypoint", ["console", "module"])
def test_installed_help_outside_project(tmp_path, entrypoint):
    command = ([console_script()] if entrypoint == "console"
               else [sys.executable, "-m", "lagrangian_fronts"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(command + ["--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--timestep" not in result.stdout


@pytest.mark.parametrize("mode", ["geographic", "cartesian"])
def test_portable_examples_through_installed_console(tmp_path, mode):
    # Copy the example so neither source-tree location nor the cwd supplies data.
    import shutil
    import pandas as pd

    project = Path(__file__).resolve().parents[1]
    source = project / "examples" / mode
    target = tmp_path / "inputs" / mode
    shutil.copytree(source, target)
    outside = tmp_path / "working"
    outside.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for config_name, run_name in (("config.yaml", mode), ("load.yaml", mode + "_loaded")):
        config_path = target / config_name
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["write_debug_outputs"] = True
        raw["plotting"]["debug_plots"] = True
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        process = subprocess.run([console_script(), "--config", str(config_path)],
                                 cwd=outside, env=env, capture_output=True, text=True)
        assert process.returncode == 0, process.stdout + process.stderr
        output = tmp_path / "outputs" / run_name
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["software"]["module_version"] == lagrangian_fronts.__version__
        assert len(list((output / "analysis/figures").glob("*.png"))) == 8
        assert pd.read_parquet(output / "analysis/flux_fronts.parquet").front_detected.any()
        assert pd.read_parquet(output / "analysis/directional_fronts.parquet").front_detected.any()
        assert (output / "matrix/transition_matrix.parquet").exists() == (config_name == "config.yaml")
        if config_name == "load.yaml":
            for table in (output / "analysis").glob("*.parquet"):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(table),
                    pd.read_parquet(tmp_path / "outputs" / mode / "analysis" / table.name),
                    check_exact=True,
                )
