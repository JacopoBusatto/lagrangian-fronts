"""Production table and reproducibility-file writing."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import __version__
from .config import AnalysisConfig

STANDARD_TABLES = (
    "cell_statistics.parquet",
    "flux_cores.parquet",
    "flux_fronts.parquet",
)
DIRECTIONAL_TABLES = (
    "directional_corridors.parquet",
    "directional_cores.parquet",
    "directional_fronts.parquet",
    "comparison.parquet",
    "component_comparison.parquet",
)


_COORDINATE_PREFIXES = (
    "start",
    "end",
    "core",
    "front",
    "candidate",
    "corridor",
    "ridge",
    "refined_core",
    "refined_axis",
    "sample",
    "flank",
    "local_gradient_max",
    "transverse_minus",
    "transverse_plus",
    "transverse_left",
    "transverse_right",
)


def _public_column_name(name: str, config: AnalysisConfig) -> str:
    """Translate a unit-neutral internal field to its stable public schema."""
    if not any(
        marker in name
        for marker in ("_length", "_rate", "_area_rate", "_rate_per_length")
    ):
        if (
            name in {"S_flux", "U_out_all", "U_out_move"}
            or name.startswith(("U_parallel_", "U_core", "U_inner", "U_outer"))
            or (
                any(
                    token in name
                    for token in (
                        "absolute_drop",
                        "absolute_flux_loss",
                        "outer_recovery",
                        "profile_total_variation",
                    )
                )
                and not name.startswith("spearman_")
            )
        ):
            name = f"{name}_rate"
        elif (
            any(
                token in name
                for token in (
                    "dS_dx",
                    "dS_dy",
                    "G_perp",
                    "G_parallel",
                    "gradient_magnitude",
                )
            )
            and not name.startswith("spearman_")
            and not name.endswith(("_ratio", "_fraction", "_percentile"))
        ):
            name = f"{name}_rate_per_length"
    replacements = (
        ("_rate_per_length", f"_{config.rate_gradient_suffix}"),
        ("_area_rate", f"_{config.area_rate_suffix}"),
        ("_rate", f"_{config.rate_suffix}"),
        ("_length", f"_{config.geometry.length_suffix}"),
    )
    for internal, public in replacements:
        name = name.replace(internal, public)

    if config.geometry.coordinate_system != "geographic":
        return name

    # Vector components retain the established east/north vocabulary in
    # geographic output, while Cartesian output uses x/y.
    vector_prefixes = (
        "U_out_",
        "D_out_",
        "mu_out_",
        "mu_in_",
        "moment_identity_",
        "directional_identity_",
    )
    if name.startswith(vector_prefixes):
        name = re.sub(r"(?<=_)x(?=_|$)", "east", name)
        name = re.sub(r"(?<=_)y(?=_|$)", "north", name)

    coordinate_names = {"x": "lon", "y": "lat", "x_bin": "lon_bin", "y_bin": "lat_bin"}
    if name in coordinate_names:
        return coordinate_names[name]
    for prefix in _COORDINATE_PREFIXES:
        if name == f"{prefix}_x":
            return f"{prefix}_lon"
        if name == f"{prefix}_y":
            return f"{prefix}_lat"
        if name == f"{prefix}_x_bin":
            return f"{prefix}_lon_bin"
        if name == f"{prefix}_y_bin":
            return f"{prefix}_lat_bin"
        if name == f"{prefix}_x_center":
            return f"{prefix}_lon_center"
        if name == f"{prefix}_y_center":
            return f"{prefix}_lat_center"
    special = {
        f"grid_x_scale_{config.geometry.length_suffix}": f"grid_zonal_scale_{config.geometry.length_suffix}",
        f"grid_y_scale_{config.geometry.length_suffix}": f"grid_meridional_scale_{config.geometry.length_suffix}",
        "x_span": "longitude_span_degrees",
        "y_span": "latitude_span_degrees",
        "centroid_x": "centroid_lon_circular",
        "centroid_y": "centroid_lat",
    }
    return special.get(name, name)


def externalize_table(table: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Return a copy with coordinate- and unit-specific public column names."""
    renamed = {name: _public_column_name(str(name), config) for name in table.columns}
    duplicates = [
        name for name in set(renamed.values()) if list(renamed.values()).count(name) > 1
    ]
    if duplicates:
        raise ValueError(f"public output column collision: {sorted(duplicates)}")
    return table.rename(columns=renamed).copy()


def _externalize_mapping(value: Any, config: AnalysisConfig) -> Any:
    if isinstance(value, dict):
        return {
            _public_column_name(str(key), config): _externalize_mapping(item, config)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_externalize_mapping(item, config) for item in value]
    if isinstance(value, tuple):
        return [_externalize_mapping(item, config) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    members = (
        sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    )
    for member in members:
        if path.is_dir():
            digest.update(member.relative_to(path).as_posix().encode("utf-8") + b"\0")
            digest.update(str(member.stat().st_size).encode("ascii") + b"\0")
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(config: AnalysisConfig) -> Path:
    output_root = Path(config.output.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / config.output.run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def write_scientific_tables(
    run_dir: Path,
    *,
    cells: pd.DataFrame,
    cores: pd.DataFrame,
    fronts: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    for name, table in zip(STANDARD_TABLES, (cells, cores, fronts), strict=True):
        externalize_table(table, config).to_parquet(run_dir / name, index=False)


def write_directional_tables(
    run_dir: Path,
    *,
    corridors: pd.DataFrame,
    cores: pd.DataFrame,
    fronts: pd.DataFrame,
    comparison: pd.DataFrame,
    component_comparison: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    tables = (corridors, cores, fronts, comparison, component_comparison)
    for name, table in zip(DIRECTIONAL_TABLES, tables, strict=True):
        externalize_table(table, config).to_parquet(run_dir / name, index=False)


def write_debug_tables(
    run_dir: Path,
    *,
    candidate_drops: pd.DataFrame,
    cross_sections: pd.DataFrame,
    section_summaries: pd.DataFrame,
    components: pd.DataFrame,
    segment_fronts: pd.DataFrame,
    config: AnalysisConfig,
) -> list[str]:
    outputs = {
        "candidate_drop_zones.parquet": candidate_drops,
        "raw_cross_sections.parquet": cross_sections,
        "section_composites.parquet": section_summaries,
        "component_graph_details.parquet": components,
        "segment_front_candidates.parquet": segment_fronts,
    }
    for name, table in outputs.items():
        externalize_table(table, config).to_parquet(run_dir / name, index=False)
    return list(outputs)


def write_directional_debug_tables(
    run_dir: Path,
    *,
    candidate_drops: pd.DataFrame,
    cross_sections: pd.DataFrame,
    section_summaries: pd.DataFrame,
    candidate_components: pd.DataFrame,
    candidate_graph_edges: pd.DataFrame,
    core_components: pd.DataFrame,
    core_graph_edges: pd.DataFrame,
    config: AnalysisConfig,
) -> list[str]:
    outputs = {
        "directional_candidate_drop_zones.parquet": candidate_drops,
        "directional_raw_cross_sections.parquet": cross_sections,
        "directional_section_composites.parquet": section_summaries,
        "directional_corridor_components.parquet": candidate_components,
        "directional_corridor_graph_edges.parquet": candidate_graph_edges,
        "directional_core_components.parquet": core_components,
        "directional_core_graph_edges.parquet": core_graph_edges,
    }
    for name, table in outputs.items():
        externalize_table(table, config).to_parquet(run_dir / name, index=False)
    return list(outputs)


def write_validation_table(
    run_dir: Path, validation: pd.DataFrame, config: AnalysisConfig
) -> str:
    name = "gradient_validation.parquet"
    externalize_table(validation, config).to_parquet(run_dir / name, index=False)
    return name


def write_reproducibility_files(
    run_dir: Path,
    *,
    config: AnalysisConfig,
    input_path: Path,
    flux_threshold_rate: float,
    counts: dict[str, int],
    transition_validation_summary: dict[str, Any],
    gradient_validation_summary: dict[str, Any] | None,
    directional_corridor_summary: dict[str, Any],
    directional_core_summary: dict[str, Any],
    directional_front_summary: dict[str, Any],
    structure_comparison_summary: dict[str, Any],
    figures: list[Path],
    optional_outputs: list[str],
    input_files: tuple[str, ...] = (),
    matrix_diagnostics: dict | None = None,
) -> None:
    config_name = "resolved_config.yaml"
    resolved_config = config.to_dict()
    resolved_config["resolved_geometry"] = config.geometry_metadata
    if config.input:
        resolved_config["resolved_input_base"] = str(
            Path(config.input_base or Path.cwd()).resolve()
        )
    (run_dir / config_name).write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    inventory = sorted(
        {
            "manifest.json",
            *(
                p.relative_to(run_dir).as_posix()
                for p in run_dir.rglob("*")
                if p.is_file()
            ),
        }
    )
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "analysis_version": config.analysis_version,
            "python": platform.python_version(),
            "module_version": __version__,
            "scientific_baseline": "10e63b4675a6b6d3f9eb1746bc9ecb6b04bf0ed5",
            "dependencies": dependency_versions(),
        },
        "config_version": config.config_version,
        "config_path": config.config_path,
        "git": git_metadata(),
        "input": {
            "path": str(input_path),
            "sha256": (matrix_diagnostics or {}).get("matrix_sha256")
            or sha256(input_path),
            "matrix_id": config.matrix.id,
            "timestep": config.matrix.timestep,
            "time_unit": config.matrix.time_unit,
            "normalized_source_probability_contract": True,
            "mode": "computed" if config.matrix.compute else "loaded",
            "resolved_files": [
                {"path": name, "sha256": sha256(Path(name))} for name in input_files
            ],
        },
        "geometry": config.geometry_metadata,
        "selection": {
            "min_moving_support": config.statistics.min_moving_support,
            "percentile": config.flux.percentile,
            "ridge_field": config.flux.ridge_field,
            f"flux_threshold_{config.rate_suffix}": flux_threshold_rate,
            "directional": {
                "minimum_P_move": config.directional.minimum_P_move,
                "minimum_R1": config.directional.minimum_R1,
                "minimum_strength": config.directional.minimum_strength,
                "maximum_neighbor_direction_difference_degrees": (
                    config.directional.maximum_neighbor_direction_difference_degrees
                ),
                "maximum_step_direction_mismatch_degrees": (
                    config.directional.maximum_step_direction_mismatch_degrees
                ),
                "minimum_component_cells": config.directional.minimum_component_cells,
                "transverse_scale_grid": config.directional.transverse_scale_grid,
                "ridge_comparison_tolerance": (
                    config.directional.ridge_comparison_tolerance
                ),
                "ridge_plateau_tolerance": (config.directional.ridge_plateau_tolerance),
                "interpolation_weight_tolerance": (
                    config.directional.interpolation_weight_tolerance
                ),
            },
        },
        "counts": counts,
        "transition_matrix_validation": _externalize_mapping(
            transition_validation_summary, config
        ),
        "gradient_validation": _externalize_mapping(
            gradient_validation_summary, config
        ),
        "directional_corridors": _externalize_mapping(
            directional_corridor_summary, config
        ),
        "directional_cores": _externalize_mapping(directional_core_summary, config),
        "directional_fronts": _externalize_mapping(directional_front_summary, config),
        "flux_directional_comparison": _externalize_mapping(
            structure_comparison_summary, config
        ),
        "output_inventory": inventory,
        "options": {
            "write_debug_outputs": config.write_debug_outputs,
            "run_validation": config.run_validation,
            "debug_plots": config.plotting.debug_plots,
        },
        "continuous_fronts_created": False,
        "topology_classification_performed": False,
        "physical_identification_performed": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, allow_nan=False), encoding="utf-8"
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dependency_versions():
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "pyarrow",
        "xarray",
        "netCDF4",
        "zarr",
        "scipy",
        "pyproj",
        "matplotlib",
        "cartopy",
        "PyYAML",
        "tqdm",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def git_metadata():
    try:
        options = {
            "cwd": Path(__file__).parent,
            "capture_output": True,
            "text": True,
            "timeout": 5,
        }
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, **options
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, **options
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return None
