"""Versioned configuration for trajectory-to-front analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ColumnConfig:
    time: str
    x: str
    y: str
    trajectory: str | None = None
    obs: str | None = None
    group_member: str | None = None


@dataclass(frozen=True)
class TrajectoryIdConfig:
    mode: str = "column"
    scope: str = "file"


@dataclass(frozen=True)
class TimeConfig:
    kind: str
    unit: str | None = None


@dataclass(frozen=True)
class InputConfig:
    paths: tuple[str, ...]
    columns: ColumnConfig
    time: TimeConfig
    format: str = "auto"
    trajectory_id: TrajectoryIdConfig = field(default_factory=TrajectoryIdConfig)


@dataclass(frozen=True)
class MatrixConfig:
    timestep: float
    time_unit: str
    compute: bool = True
    path: str | None = None
    id: str | None = None
    segment_mode: str = "non_overlapping"
    max_interpolation_gap: float | None = None


@dataclass(frozen=True)
class OutputConfig:
    root: str
    run_name: str = "lagrangian_currents"


LENGTH_UNITS_TO_METERS = {
    "mm": 1.0e-3,
    "cm": 1.0e-2,
    "m": 1.0,
    "km": 1.0e3,
}
TIME_UNITS = ("s", "min", "h", "day")


@dataclass(frozen=True)
class SpatialGeometryConfig:
    coordinate_system: str
    length_unit: str
    ellipsoid: str | None = None

    @property
    def length_suffix(self) -> str:
        return self.length_unit

    def rate_suffix(self, time_unit: str) -> str:
        return f"{self.length_unit}_{time_unit}"

    def area_rate_suffix(self, time_unit: str) -> str:
        return f"{self.length_unit}2_{time_unit}"

    def rate_gradient_suffix(self, time_unit: str) -> str:
        return f"{self.length_unit}_{time_unit}_per_{self.length_unit}"


@dataclass(frozen=True)
class GridConfig:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    dlon: float
    dlat: float
    periodic_longitude: bool = True

    @property
    def nlon(self) -> int:
        return round((self.lon_max - self.lon_min) / self.dlon)

    @property
    def nlat(self) -> int:
        return round((self.lat_max - self.lat_min) / self.dlat)

    @property
    def x_min(self) -> float:
        return self.lon_min

    @property
    def x_max(self) -> float:
        return self.lon_max

    @property
    def y_min(self) -> float:
        return self.lat_min

    @property
    def y_max(self) -> float:
        return self.lat_max

    @property
    def dx(self) -> float:
        return self.dlon

    @property
    def dy(self) -> float:
        return self.dlat

    @property
    def nx(self) -> int:
        return self.nlon

    @property
    def ny(self) -> int:
        return self.nlat

    @property
    def periodic_x(self) -> bool:
        return self.periodic_longitude


@dataclass(frozen=True)
class CartesianGridConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float

    @property
    def nx(self) -> int:
        return round((self.x_max - self.x_min) / self.dx)

    @property
    def ny(self) -> int:
        return round((self.y_max - self.y_min) / self.dy)

    @property
    def periodic_x(self) -> bool:
        return False


@dataclass(frozen=True)
class StatisticsConfig:
    min_moving_support: int = 10
    angular_bins: int = 36
    direction_zero_tolerance: float = 1.0e-12
    high_R1: float = 0.8
    low_R1: float = 0.5


@dataclass(frozen=True)
class FluxConfig:
    percentile: float = 0.9
    ridge_field: str = "raw"
    transverse_scale_grid: float = 1.0
    interpolation_weight_tolerance: float = 1.0e-10
    orientation_reliable_R1: float = 0.8
    orientation_ambiguous_R1: float = 0.5
    direction_disagreement_degrees: float = 20.0
    abrupt_tangent_mismatch_degrees: float = 45.0
    ridge_comparison_tolerance: float = 1.0e-12
    smoothing_window_cells: int = 3


@dataclass(frozen=True)
class DirectionalConfig:
    minimum_P_move: float = 0.5
    minimum_R1: float = 0.8
    minimum_strength: float = 0.5
    maximum_neighbor_direction_difference_degrees: float = 45.0
    maximum_step_direction_mismatch_degrees: float = 45.0
    minimum_component_cells: int = 3
    transverse_scale_grid: float = 1.0
    ridge_comparison_tolerance: float = 1.0e-12
    ridge_plateau_tolerance: float = 1.0e-12
    interpolation_weight_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class FrontConfig:
    half_width_grid_scales: int = 5
    sampling_interval_grid_scales: float = 1.0
    core_refinement_grid_scales: float = 1.0
    robust_median_window_samples: int = 3
    composite_half_window_sections: int = 2
    minimum_persistent_neighbor_sections: int = 2
    minimum_persistent_fraction: float = 0.5
    diagnostic_low_R1: float = 0.5
    diagnostic_large_direction_disagreement_degrees: float = 20.0
    diagnostic_high_curvature_degrees: float = 60.0
    diagnostic_strong_outer_recovery_fraction: float = 0.5
    diagnostic_min_full_section_valid_samples: int = 7
    nearby_branch_cross_distance_scales: float = 5.0
    nearby_branch_along_distance_scales: float = 1.0


@dataclass(frozen=True)
class ValidationConfig:
    normalization_atol: float = 1.0e-12
    probability_rtol: float = 1.0e-10
    probability_atol: float = 1.0e-12
    center_atol: float = 1.0e-9
    gradient_zero_tolerance: float = 1.0e-12
    interpolation_weight_tolerance: float = 1.0e-10
    gradient_search_radius_grid_scales: float = 1.0
    local_background_radius_grid_scales: float = 2.0
    duplicate_disagreement_grid_scales: float = 1.0
    core_gradient_ratio_epsilon: float = 1.0e-12
    multiple_drop_similarity_fraction: float = 0.1
    direct_sample_atol_grid_cells: float = 1.0e-8


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    dpi: int = 160
    projection: str = "SouthPolarStereo"
    central_longitude: float = 0.0
    circular_boundary: bool = True
    draw_coastlines: bool = True
    vector_stride_cells: int = 5
    vector_reference: float = 5.0
    directional_vector_reference: float = 0.5
    structure_map_max_percentile: float = 100.0
    debug_plots: bool = False


@dataclass(frozen=True)
class AnalysisConfig:
    matrix: MatrixConfig
    output: OutputConfig
    geometry: SpatialGeometryConfig
    grid: GridConfig | CartesianGridConfig
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    fronts: FrontConfig = field(default_factory=FrontConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
    write_debug_outputs: bool = False
    run_validation: bool = False
    analysis_version: str = "5.0.0-production"
    config_version: int = 1
    input: InputConfig | None = None
    config_path: str | None = field(default=None, repr=False, compare=False)
    input_base: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_types(self)
        if type(self.config_version) is not int or self.config_version != 1:
            raise ValueError("unsupported config_version; expected 1")
        if not _positive(self.matrix.timestep):
            raise ValueError("matrix.timestep must be finite and positive")
        if self.matrix.time_unit not in TIME_UNITS:
            raise ValueError(f"unsupported time_unit: {self.matrix.time_unit}")
        seconds_per_unit = {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0}
        if not math.isfinite(
            self.matrix.timestep * seconds_per_unit[self.matrix.time_unit]
        ):
            raise ValueError("matrix.timestep overflows elapsed seconds")
        if self.matrix.segment_mode not in {"non_overlapping", "every_observation"}:
            raise ValueError(
                "matrix.segment_mode must be non_overlapping or every_observation"
            )
        gap = self.matrix.max_interpolation_gap
        if gap is not None and not _positive(gap):
            raise ValueError(
                "matrix.max_interpolation_gap must be null or finite and positive"
            )
        if gap is not None and not math.isfinite(
            gap * seconds_per_unit[self.matrix.time_unit]
        ):
            raise ValueError("matrix.max_interpolation_gap overflows elapsed seconds")
        if not self.matrix.compute and not self.matrix.path:
            raise ValueError("matrix.path is required when matrix.compute is false")
        if self.matrix.compute and self.matrix.path is not None:
            raise ValueError("matrix.path must be null when matrix.compute is true")
        if self.matrix.id is None:
            object.__setattr__(
                self, "matrix", replace(self.matrix, id=self.output.run_name)
            )
        if not self.matrix.id.strip():
            raise ValueError("matrix.id must not be blank")
        if not self.output.root.strip():
            raise ValueError("output.root must not be blank")
        if (
            not self.output.run_name.strip()
            or self.output.run_name in {".", ".."}
            or any(c in self.output.run_name for c in "/\\")
        ):
            raise ValueError("output.run_name must be a single directory name")
        if self.matrix.compute and self.input is None:
            raise ValueError("input is required when matrix.compute is true")
        if self.input is not None:
            inp = self.input
            if (
                not isinstance(inp.paths, (tuple, list))
                or not inp.paths
                or any(not isinstance(p, str) or not p.strip() for p in inp.paths)
            ):
                raise ValueError("input.paths must contain one or more path strings")
            if inp.format not in {"auto", "csv", "parquet", "netcdf", "zarr"}:
                raise ValueError("unsupported input.format")
            if inp.trajectory_id.mode not in {
                "column",
                "filename",
            } or inp.trajectory_id.scope not in {"file", "global"}:
                raise ValueError("invalid input.trajectory_id mode or scope")
            if inp.trajectory_id.mode == "column" and not inp.columns.trajectory:
                raise ValueError("input.columns.trajectory is required in column mode")
            names = [v for v in asdict(inp.columns).values() if v is not None]
            if any(not v.strip() for v in names) or len(set(names)) != len(names):
                raise ValueError("input.columns mappings must be nonempty and distinct")
            if inp.time.kind not in {"datetime", "numeric"}:
                raise ValueError("input.time.kind must be datetime or numeric")
            if inp.time.kind == "numeric" and inp.time.unit not in TIME_UNITS:
                raise ValueError("numeric input.time.unit must be s, min, h, or day")
            if inp.time.kind == "datetime" and inp.time.unit is not None:
                raise ValueError("datetime input.time must not specify unit")
        if self.geometry.coordinate_system not in {"geographic", "cartesian"}:
            raise ValueError(
                "geometry.coordinate_system must be 'geographic' or 'cartesian'"
            )
        if self.geometry.length_unit not in LENGTH_UNITS_TO_METERS:
            raise ValueError(f"unsupported length_unit: {self.geometry.length_unit}")
        if self.geometry.coordinate_system == "geographic":
            if not isinstance(self.grid, GridConfig):
                raise ValueError(
                    "geographic geometry requires a longitude/latitude grid"
                )
            if not self.geometry.ellipsoid or not self.geometry.ellipsoid.strip():
                raise ValueError("geographic geometry requires geometry.ellipsoid")
        else:
            if not isinstance(self.grid, CartesianGridConfig):
                raise ValueError("cartesian geometry requires an x/y grid")
            if self.geometry.ellipsoid is not None:
                raise ValueError("cartesian geometry must not define an ellipsoid")
        grid_values = (
            self.grid.x_min,
            self.grid.x_max,
            self.grid.y_min,
            self.grid.y_max,
            self.grid.dx,
            self.grid.dy,
        )
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in grid_values
        ):
            raise ValueError("grid bounds and spacing must be finite numbers")
        if min(self.grid.dx, self.grid.dy) <= 0:
            raise ValueError("grid spacing must be positive")
        if self.grid.x_max <= self.grid.x_min or self.grid.y_max <= self.grid.y_min:
            raise ValueError("grid maxima must exceed minima")
        for axis, span, spacing in (
            ("x", self.grid.x_max - self.grid.x_min, self.grid.dx),
            ("y", self.grid.y_max - self.grid.y_min, self.grid.dy),
        ):
            count = span / spacing
            if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"grid {axis} span must be an integer multiple of its spacing"
                )
        if isinstance(self.grid, GridConfig):
            if not -90.0 <= self.grid.lat_min < self.grid.lat_max <= 90.0:
                raise ValueError(
                    "geographic grid latitude bounds must lie in [-90, 90]"
                )
            if not isinstance(self.grid.periodic_longitude, bool):
                raise TypeError("grid.periodic_longitude must be boolean")
        if self.statistics.min_moving_support <= 0:
            raise ValueError("min_moving_support must be positive")
        if self.statistics.angular_bins < 4:
            raise ValueError("angular_bins must be at least four")
        if not 0 <= self.statistics.low_R1 <= self.statistics.high_R1 <= 1:
            raise ValueError("R1 diagnostic thresholds must be ordered in [0, 1]")
        if self.statistics.direction_zero_tolerance < 0:
            raise ValueError("direction_zero_tolerance must be nonnegative")
        if not 0 < self.flux.percentile < 1:
            raise ValueError("percentile must lie in (0, 1)")
        if self.flux.ridge_field not in {"raw", "smoothed"}:
            raise ValueError("ridge_field must be 'raw' or 'smoothed'")
        if self.flux.transverse_scale_grid <= 0:
            raise ValueError("transverse_scale_grid must be positive")
        if self.flux.smoothing_window_cells != 3:
            raise ValueError("the validated optional smoothing window is 3x3")
        if (
            min(
                self.flux.interpolation_weight_tolerance,
                self.flux.ridge_comparison_tolerance,
            )
            < 0
        ):
            raise ValueError("branch numerical tolerances must be nonnegative")
        if not (
            0
            <= self.flux.orientation_ambiguous_R1
            <= self.flux.orientation_reliable_R1
            <= 1
        ):
            raise ValueError(
                "branch R1 diagnostic thresholds must be ordered in [0, 1]"
            )
        if any(
            not 0 <= value <= 180
            for value in (
                self.flux.direction_disagreement_degrees,
                self.flux.abrupt_tangent_mismatch_degrees,
            )
        ):
            raise ValueError("branch angular diagnostics must lie in [0, 180]")
        if any(
            not 0 <= value <= 1
            for value in (
                self.directional.minimum_P_move,
                self.directional.minimum_R1,
                self.directional.minimum_strength,
            )
        ):
            raise ValueError(
                "directional probability/strength thresholds must be in [0, 1]"
            )
        if any(
            not 0 <= value <= 90
            for value in (
                self.directional.maximum_neighbor_direction_difference_degrees,
                self.directional.maximum_step_direction_mismatch_degrees,
            )
        ):
            raise ValueError("directional local-angle thresholds must be in [0, 90]")
        if self.directional.minimum_component_cells < 1:
            raise ValueError("minimum_component_cells must be positive")
        if self.directional.transverse_scale_grid <= 0:
            raise ValueError("directional transverse scale must be positive")
        if (
            min(
                self.directional.ridge_comparison_tolerance,
                self.directional.ridge_plateau_tolerance,
                self.directional.interpolation_weight_tolerance,
            )
            < 0
        ):
            raise ValueError("directional numerical tolerances must be nonnegative")
        if self.fronts.half_width_grid_scales < 2:
            raise ValueError("cross-section half-width must be at least two cells")
        if self.fronts.sampling_interval_grid_scales <= 0:
            raise ValueError("cross-section sampling interval must be positive")
        if not 0 < self.fronts.core_refinement_grid_scales <= 1:
            raise ValueError("core refinement must be within one grid scale")
        if (
            self.fronts.robust_median_window_samples < 1
            or self.fronts.robust_median_window_samples % 2 == 0
        ):
            raise ValueError("robust median window must be a positive odd integer")
        if self.fronts.composite_half_window_sections < 0:
            raise ValueError("composite half-width must be nonnegative")
        if self.fronts.minimum_persistent_neighbor_sections < 1:
            raise ValueError("minimum persistent neighbors must be positive")
        if not 0 <= self.fronts.minimum_persistent_fraction <= 1:
            raise ValueError("minimum persistent fraction must lie in [0, 1]")
        if (
            min(
                self.validation.normalization_atol,
                self.validation.probability_rtol,
                self.validation.probability_atol,
                self.validation.center_atol,
                self.validation.gradient_zero_tolerance,
                self.validation.interpolation_weight_tolerance,
                self.validation.gradient_search_radius_grid_scales,
                self.validation.local_background_radius_grid_scales,
                self.validation.direct_sample_atol_grid_cells,
            )
            < 0
        ):
            raise ValueError("validation tolerances and radii must be nonnegative")
        if self.plotting.projection not in {"PlateCarree", "SouthPolarStereo"}:
            raise ValueError("unsupported plotting projection")
        if self.plotting.dpi <= 0:
            raise ValueError("plotting dpi must be positive")
        if self.plotting.vector_stride_cells < 1:
            raise ValueError("vector_stride_cells must be positive")
        if self.plotting.vector_reference <= 0:
            raise ValueError("vector_reference must be positive")
        if not 0 < self.plotting.directional_vector_reference <= 1:
            raise ValueError("directional_vector_reference must lie in (0, 1]")
        if not 0 < self.plotting.structure_map_max_percentile <= 100:
            raise ValueError("structure_map_max_percentile must lie in (0, 100]")
        if self.plotting.debug_plots and not self.run_validation:
            raise ValueError("debug validation plots require run_validation=true")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("config_path")
        data.pop("input_base")
        if data["input"] is not None:
            data["input"]["paths"] = list(data["input"]["paths"])
        if self.geometry.coordinate_system == "cartesian":
            data["geometry"].pop("ellipsoid", None)
        return data

    @property
    def rate_unit(self) -> str:
        return f"{self.geometry.length_unit}/{self.matrix.time_unit}"

    @property
    def rate_suffix(self) -> str:
        return self.geometry.rate_suffix(self.matrix.time_unit)

    @property
    def area_rate_suffix(self) -> str:
        return self.geometry.area_rate_suffix(self.matrix.time_unit)

    @property
    def rate_gradient_suffix(self) -> str:
        return self.geometry.rate_gradient_suffix(self.matrix.time_unit)

    @property
    def geometry_metadata(self) -> dict[str, str]:
        return {
            "coordinate_system": self.geometry.coordinate_system,
            "coordinate_unit": (
                "degree"
                if self.geometry.coordinate_system == "geographic"
                else self.geometry.length_unit
            ),
            "length_unit": self.geometry.length_unit,
            "time_unit": self.matrix.time_unit,
            "rate_unit": self.rate_unit,
            "bearing_convention": (
                "degrees clockwise from positive y/north; 0=+y/north, 90=+x/east"
            ),
            "geometry_backend": (
                f"pyproj.Geod({self.geometry.ellipsoid})"
                if self.geometry.coordinate_system == "geographic"
                else "Euclidean planar"
            ),
        }


def _construct(cls, values: Mapping[str, Any], section: str):
    if not isinstance(values, Mapping):
        raise TypeError(f"{section} must be a YAML mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} keys: {', '.join(unknown)}")
    try:
        return cls(**dict(values))
    except TypeError as exc:
        raise ValueError(f"invalid {section} configuration: {exc}") from exc


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_types(instance):
    """Check scalar types before range comparisons, including nested dataclasses."""
    from dataclasses import is_dataclass
    from typing import get_args, get_origin, get_type_hints

    for item, annotation in get_type_hints(type(instance)).items():
        value = getattr(instance, item)
        if is_dataclass(value):
            _validate_types(value)
            continue
        import types
        from typing import Union

        alternatives = (
            get_args(annotation)
            if get_origin(annotation) in (types.UnionType, Union)
            else (annotation,)
        )
        if value is None and type(None) in alternatives:
            continue
        scalar = [t for t in alternatives if t in (bool, int, float, str)]
        if scalar:
            valid = any(
                type(value) is t or (t is float and type(value) is int) for t in scalar
            )
            if not valid:
                raise ValueError(
                    f"{type(instance).__name__}.{item}: invalid value type"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{item} must be finite")


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("configuration root must be a YAML mapping")
    if type(raw.get("config_version")) is not int or raw["config_version"] != 1:
        raise ValueError(
            "unsupported or missing config_version; expected 1. Migrate input matrix fields to matrix, branches to flux, and edges to fronts."
        )
    if "transport" in raw:
        raise ValueError(
            "configuration section 'transport' was renamed to 'flux'; "
            "replace transport: with flux: and preserve its values. "
            "Compatibility aliases are not supported."
        )
    allowed = {f.name for f in fields(AnalysisConfig)} - {"config_path", "input_base"}
    unknown = set(raw) - allowed - {"resolved_geometry", "resolved_input_base"}
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    missing = {"matrix", "output", "geometry", "grid"} - set(raw)
    if missing:
        raise ValueError(
            f"missing configuration sections: {', '.join(sorted(missing))}"
        )
    geometry = _construct(SpatialGeometryConfig, raw["geometry"], "geometry")
    if geometry.coordinate_system not in {"geographic", "cartesian"}:
        raise ValueError("geometry.coordinate_system must be geographic or cartesian")
    grid_type = (
        GridConfig
        if geometry.coordinate_system == "geographic"
        else CartesianGridConfig
    )
    values = dict(raw)
    for key in ("resolved_geometry", "resolved_input_base"):
        values.pop(key, None)
    values["geometry"] = geometry
    section_types = {
        "grid": grid_type,
        "matrix": MatrixConfig,
        "output": OutputConfig,
        "statistics": StatisticsConfig,
        "flux": FluxConfig,
        "directional": DirectionalConfig,
        "fronts": FrontConfig,
        "validation": ValidationConfig,
        "plotting": PlotConfig,
    }
    for section, cls in section_types.items():
        values[section] = _construct(cls, raw.get(section, {}), section)
    if raw.get("input") is not None:
        inp = raw["input"]
        if not isinstance(inp, Mapping):
            raise ValueError("input must be a YAML mapping")
        inp = dict(inp)
        for key, cls in (
            ("columns", ColumnConfig),
            ("time", TimeConfig),
            ("trajectory_id", TrajectoryIdConfig),
        ):
            inp[key] = _construct(cls, inp.get(key, {}), f"input.{key}")
        if isinstance(inp.get("paths"), list):
            inp["paths"] = tuple(inp["paths"])
        values["input"] = _construct(InputConfig, inp, "input")
    config = AnalysisConfig(
        **values, config_path=str(config_path), input_base=str(config_path.parent)
    )

    def resolve(value):
        candidate = Path(value).expanduser()
        return str((config_path.parent / candidate).resolve())

    config = replace(
        config, output=replace(config.output, root=resolve(config.output.root))
    )
    if config.matrix.path:
        config = replace(
            config, matrix=replace(config.matrix, path=resolve(config.matrix.path))
        )
    if config.input:
        config = replace(
            config,
            input=replace(
                config.input, paths=tuple(resolve(p) for p in config.input.paths)
            ),
        )
    if "resolved_input_base" in raw:
        config = replace(config, input_base=resolve(raw["resolved_input_base"]))
    if (
        "resolved_geometry" in raw
        and raw["resolved_geometry"] != config.geometry_metadata
    ):
        raise ValueError("resolved_geometry metadata does not match the configuration")
    return config
