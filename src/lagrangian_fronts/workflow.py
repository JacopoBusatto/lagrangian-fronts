"""Single-YAML orchestration and output writing."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import yaml

from .analysis import _stage, analyze_transition_matrix
from .config import AnalysisConfig
from .io import (
    create_run_directory,
    json_safe,
    sha256,
    write_debug_tables,
    write_directional_debug_tables,
    write_directional_tables,
    write_reproducibility_files,
    write_scientific_tables,
    write_validation_table,
)
from .matrix import compute_transition_matrix, matrix_metadata, read_transition_matrix
from .plotting import create_standard_figures
from .statistics import compact_cell_table
from .trajectories import read_trajectories


def run(config: AnalysisConfig) -> Path:
    """Execute compute/load and the identical downstream analysis once."""
    config = replace(
        config,
        output=replace(
            config.output, root=str(Path(config.output.root).expanduser().resolve())
        ),
        input_base=str(Path(config.input_base or Path.cwd()).resolve()),
    )
    if config.matrix.path:
        config = replace(
            config,
            matrix=replace(
                config.matrix, path=str(Path(config.matrix.path).expanduser().resolve())
            ),
        )
    run_dir = create_run_directory(config)
    try:
        _run_into(config, run_dir)
    except Exception as exc:
        manifest = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "config_path": config.config_path,
            "output_inventory": sorted(
                p.relative_to(run_dir).as_posix()
                for p in run_dir.rglob("*")
                if p.is_file()
            ),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        raise
    return run_dir


def _run_into(config, run_dir):
    analysis_dir = run_dir / "analysis"
    matrix_dir = run_dir / "matrix"
    analysis_dir.mkdir()
    matrix_dir.mkdir()
    input_files = ()
    matrix_summary = {}
    if config.matrix.compute:
        trajectories = _stage("Reading trajectories", read_trajectories, config)
        input_files = trajectories.input_files
        config = replace(config, input=replace(config.input, paths=input_files))
        built = _stage("Building matrix", compute_transition_matrix, trajectories, config)
        matrix = built.transition_table
        matrix_summary = built.diagnostics
        input_path = matrix_dir / "transition_matrix.parquet"
        matrix.to_parquet(input_path, index=False)
        del trajectories
    else:
        input_path = Path(config.matrix.path).resolve()
        matrix = _stage("Reading matrix", read_transition_matrix, input_path, config)
        sidecar = input_path.with_name("matrix_summary.yaml")
        if sidecar.exists():
            matrix_summary = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        else:
            matrix_summary = {
                **matrix_metadata(config),
                **{
                    key: None
                    for key in (
                        "segment_mode",
                        "max_interpolation_gap",
                        "source_timestep_seconds",
                        "number_of_resolved_input_files",
                        "source_trajectories",
                        "valid_observations",
                        "structurally_rejected_observations",
                        "discarded_short_trajectories",
                        "candidate_transitions",
                        "exact_endpoints",
                        "interpolated_endpoints",
                        "missing_endpoints",
                        "gap_rejected_endpoints",
                        "rejected_missing_endpoint",
                        "rejected_interpolation_gap",
                        "rejected_outside_domain",
                    )
                },
            }
    logging.getLogger("lagrangian_fronts").info("Writing matrix metadata...")
    matrix_summary = {
        **matrix_summary,
        "summary_version": 1,
        "matrix_sha256": sha256(input_path),
        "mode": "computed" if config.matrix.compute else "loaded",
        "matrix_path": str(input_path),
    }
    summary_path = matrix_dir / "matrix_summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(json_safe(matrix_summary), sort_keys=False), encoding="utf-8"
    )
    resolved = config.to_dict()
    if config.input:
        resolved["resolved_input_base"] = str(
            Path(config.input_base or Path.cwd()).resolve()
        )
    resolved["resolved_geometry"] = config.geometry_metadata
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    result = analyze_transition_matrix(matrix, config)
    statistics, cores, fronts = (
        result.statistics,
        result.flux_cores,
        result.flux_fronts,
    )
    directional_corridors, directional_cores = (
        result.directional_corridors,
        result.directional_cores,
    )
    directional_fronts, comparison, validation = (
        result.directional_fronts,
        result.comparison,
        result.validation,
    )
    matrix_summary.update(statistics.validation_summary)
    matrix_summary["retained_in_domain_transitions"] = int(
        matrix.transition_count.sum()
    )
    summary_path.write_text(
        yaml.safe_dump(json_safe(matrix_summary), sort_keys=False), encoding="utf-8"
    )
    logging.getLogger("lagrangian_fronts").info("Writing analysis tables...")
    cells = compact_cell_table(statistics.cells)
    write_scientific_tables(
        analysis_dir,
        cells=cells,
        cores=cores.cores,
        fronts=fronts.fronts,
        config=config,
    )
    write_directional_tables(
        analysis_dir,
        corridors=directional_cores.candidate_diagnostics,
        cores=directional_cores.cores,
        fronts=directional_fronts.fronts,
        comparison=comparison.cells,
        component_comparison=comparison.components,
        config=config,
    )

    optional_outputs: list[str] = []
    if config.write_debug_outputs:
        optional_outputs.extend(
            write_debug_tables(
                analysis_dir,
                candidate_drops=fronts.candidate_drops,
                cross_sections=fronts.cross_sections,
                section_summaries=fronts.section_summaries,
                components=cores.components,
                segment_fronts=fronts.segment_fronts,
                config=config,
            )
        )
        optional_outputs.extend(
            write_directional_debug_tables(
                analysis_dir,
                candidate_drops=directional_fronts.candidate_drops,
                cross_sections=directional_fronts.cross_sections,
                section_summaries=directional_fronts.section_summaries,
                candidate_components=directional_corridors.components,
                candidate_graph_edges=directional_corridors.edges,
                core_components=directional_cores.components,
                core_graph_edges=directional_cores.edges,
                config=config,
            )
        )
    if validation is not None:
        optional_outputs.append(
            write_validation_table(analysis_dir, validation.validation, config)
        )

    figures = _stage(
        "Figures", create_standard_figures,
        statistics.cells,
        cores,
        fronts,
        config,
        analysis_dir / "figures",
        directional_cores=directional_cores,
        directional_fronts=directional_fronts,
        validation=validation,
    )
    status_counts = fronts.fronts.front_status.value_counts()
    directional_status_counts = directional_fronts.fronts.front_status.value_counts()
    _stage(
        "Finalizing run", write_reproducibility_files,
        run_dir,
        config=config,
        input_path=input_path,
        flux_threshold_rate=cores.threshold_rate,
        counts={
            "cells": len(cells),
            "current_core_cells": len(cores.cores),
            "core_sides": len(fronts.fronts),
            "probable_flux_fronts": int(
                status_counts.get("probable_flux_front", 0)
            ),
            "observable_sides_without_front": int(
                status_counts.get("observable_no_retained_front", 0)
            ),
            "unobservable_sides": int(status_counts.get("side_not_observable", 0)),
            "directional_corridor_cells": len(directional_corridors.corridors),
            "directional_corridor_components": len(directional_corridors.components),
            "directional_core_cells": len(directional_cores.cores),
            "directional_core_components": len(directional_cores.components),
            "directional_sides": len(directional_fronts.fronts),
            "probable_directional_fronts": int(
                directional_status_counts.get("probable_directional_front", 0)
            ),
            "observable_sides_without_directional_front": int(
                directional_status_counts.get(
                    "observable_no_retained_directional_front", 0
                )
            ),
            "unobservable_directional_sides": int(
                directional_status_counts.get("side_not_observable", 0)
            ),
        },
        transition_validation_summary=statistics.validation_summary,
        gradient_validation_summary=(validation.summary if validation else None),
        directional_corridor_summary=directional_corridors.summary,
        directional_core_summary=directional_cores.summary,
        directional_front_summary=directional_fronts.summary,
        structure_comparison_summary=comparison.summary,
        figures=figures,
        optional_outputs=optional_outputs,
        input_files=input_files,
        matrix_diagnostics=matrix_summary,
    )
