"""Self-contained trajectory, transition-matrix, and Lagrangian front research."""

from .analysis import AnalysisResult, analyze_transition_matrix
from .config import AnalysisConfig, load_config
from .matrix import MatrixResult, compute_transition_matrix, read_transition_matrix
from .trajectories import TrajectoryData, read_trajectories

__version__ = "0.1.0"


def run(config):
    """Load plotting and output dependencies only when executing a full run."""
    from .workflow import run as execute

    return execute(config)


__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "MatrixResult",
    "TrajectoryData",
    "analyze_transition_matrix",
    "compute_transition_matrix",
    "load_config",
    "read_trajectories",
    "read_transition_matrix",
    "run",
]
