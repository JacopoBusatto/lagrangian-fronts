import os
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from lagrangian_fronts import config as c

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/lagrangian_fronts"


def test_every_public_yaml_parameter_is_documented():
    text = (ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    sections = {
        "input": c.InputConfig,
        "input.columns": c.ColumnConfig,
        "input.time": c.TimeConfig,
        "input.trajectory_id": c.TrajectoryIdConfig,
        "matrix": c.MatrixConfig,
        "geometry": c.SpatialGeometryConfig,
        "grid": c.GridConfig,
        "statistics": c.StatisticsConfig,
        "flux": c.FluxConfig,
        "directional": c.DirectionalConfig,
        "fronts": c.FrontConfig,
        "validation": c.ValidationConfig,
        "output": c.OutputConfig,
        "plotting": c.PlotConfig,
    }
    for prefix, cls in [*sections.items(), ("grid", c.CartesianGridConfig)]:
        for item in fields(cls):
            assert f"| `{prefix}.{item.name}` |" in text
    for key in (
        "config_version",
        "analysis_version",
        "run_validation",
        "write_debug_outputs",
    ):
        assert f"| `{key}` |" in text


def test_module_imports_after_extraction_without_repository(tmp_path):
    target = tmp_path / "lagrangian_fronts"
    target.mkdir()
    for source in MODULE.glob("*.py"):
        shutil.copyfile(source, target / source.name)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('kinematicparcels', 'research')):
        raise AssertionError('repository dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import lagrangian_fronts
from lagrangian_fronts.io import git_metadata
assert callable(lagrangian_fronts.analyze_transition_matrix)
assert git_metadata() is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
