import pytest

from lagrangian_fronts.config import (
    AnalysisConfig,
    CartesianGridConfig,
    ColumnConfig,
    InputConfig,
    MatrixConfig,
    OutputConfig,
    PlotConfig,
    SpatialGeometryConfig,
    StatisticsConfig,
    TimeConfig,
)


@pytest.fixture
def config(tmp_path):
    return AnalysisConfig(
        matrix=MatrixConfig(timestep=2.0, time_unit="s"),
        input=InputConfig(
            paths=(str(tmp_path / "tracks.csv"),),
            columns=ColumnConfig(trajectory="track", time="t", x="X", y="Y"),
            time=TimeConfig("numeric", "s"),
        ),
        output=OutputConfig(str(tmp_path / "outputs"), "test"),
        geometry=SpatialGeometryConfig("cartesian", "m"),
        grid=CartesianGridConfig(0.0, 4.0, 0.0, 4.0, 1.0, 1.0),
        statistics=StatisticsConfig(min_moving_support=1),
        plotting=PlotConfig(enabled=False),
        input_base=str(tmp_path),
    )
