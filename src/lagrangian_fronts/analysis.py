"""The frozen scientific pipeline: dataframe in, scientific results out."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter

import pandas as pd

from .comparison import (
    StructureComparison,
    compare_flux_and_directional_structures,
)
from .config import AnalysisConfig
from .cores import CoreSolution, compute_current_cores
from .directional_cores import DirectionalCoreSolution, compute_directional_cores
from .directional_corridors import (
    DirectionalCorridorSolution,
    compute_directional_corridors,
)
from .directional_fronts import (
    DirectionalFrontSolution,
    compute_probable_directional_fronts,
)
from .fronts import FrontSolution, compute_probable_fronts
from .statistics import TransitionStatistics, compute_transition_statistics
from .validation import ValidationSolution, compute_validation


def _stage(name, function, *args, **kwargs):
    """Log a timed operation; the CLI enables this package's logger."""
    logger = logging.getLogger("lagrangian_fronts")
    logger.info("%s...", name)
    started = perf_counter()
    result = function(*args, **kwargs)
    logger.info("%s completed in %.1f s", name, perf_counter() - started)
    return result


@dataclass(frozen=True)
class AnalysisResult:
    statistics: TransitionStatistics
    directional_corridors: DirectionalCorridorSolution
    directional_cores: DirectionalCoreSolution
    directional_fronts: DirectionalFrontSolution
    flux_cores: CoreSolution
    flux_fronts: FrontSolution
    comparison: StructureComparison
    validation: ValidationSolution | None


def analyze_transition_matrix(
    table: pd.DataFrame, config: AnalysisConfig
) -> AnalysisResult:
    """Analyze a matrix with no trajectory, filesystem, or plotting dependency."""
    matrix = table
    statistics = _stage("Statistics", compute_transition_statistics, matrix, config)
    directional_corridors = _stage(
        "Directional corridors", compute_directional_corridors, statistics.cells, config
    )
    directional_cores = _stage(
        "Directional cores", compute_directional_cores,
        statistics.cells, directional_corridors, config
    )
    directional_fronts = _stage(
        "Directional fronts", compute_probable_directional_fronts,
        statistics.cells, directional_cores, config
    )
    cores = _stage("Flux cores", compute_current_cores, statistics.cells, config)
    fronts = _stage("Flux fronts", compute_probable_fronts, statistics.cells, cores, config)
    comparison = _stage(
        "Comparison", compare_flux_and_directional_structures,
        statistics.cells, cores, directional_cores, config
    )
    validation = (
        _stage("Validation", compute_validation, statistics.cells, fronts, config)
        if config.run_validation
        else None
    )
    return AnalysisResult(
        statistics,
        directional_corridors,
        directional_cores,
        directional_fronts,
        cores,
        fronts,
        comparison,
        validation,
    )
