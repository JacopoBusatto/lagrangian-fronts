"""Shared grid, angle, interpolation, and physical-scale geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pyproj import Geod

from .config import LENGTH_UNITS_TO_METERS, SpatialGeometryConfig

NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class SpatialGeometry(Protocol):
    """Geometry operations in one configured physical length unit."""

    coordinate_system: str
    length_unit: str

    def inverse(self, x1, y1, x2, y2):
        """Return forward bearing, reverse bearing, and physical distance."""

    def forward(self, x, y, bearing, distance):
        """Advance by a physical distance and return x, y, reverse bearing."""


@dataclass(frozen=True)
class GeographicGeometry:
    ellipsoid: str
    length_unit: str
    coordinate_system: str = "geographic"
    _backend: Geod = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_backend", Geod(ellps=self.ellipsoid))

    @property
    def _meters_per_unit(self) -> float:
        return LENGTH_UNITS_TO_METERS[self.length_unit]

    def inverse(self, x1, y1, x2, y2):
        forward, reverse, distance_m = self._backend.inv(x1, y1, x2, y2)
        return forward, reverse, np.asarray(distance_m) / self._meters_per_unit

    def forward(self, x, y, bearing, distance):
        return self._backend.fwd(
            x,
            y,
            bearing,
            np.asarray(distance) * self._meters_per_unit,
        )


@dataclass(frozen=True)
class CartesianGeometry:
    length_unit: str
    coordinate_system: str = "cartesian"

    @staticmethod
    def _restore_scalar(value: np.ndarray, scalar: bool):
        return float(value) if scalar else value

    def inverse(self, x1, y1, x2, y2):
        scalar = all(np.ndim(value) == 0 for value in (x1, y1, x2, y2))
        delta_x = np.asarray(x2, dtype=float) - np.asarray(x1, dtype=float)
        delta_y = np.asarray(y2, dtype=float) - np.asarray(y1, dtype=float)
        distance = np.hypot(delta_x, delta_y)
        forward = np.remainder(np.rad2deg(np.arctan2(delta_x, delta_y)), 360.0)
        reverse = np.remainder(forward + 180.0, 360.0)
        return tuple(
            self._restore_scalar(np.asarray(value), scalar)
            for value in (forward, reverse, distance)
        )

    def forward(self, x, y, bearing, distance):
        scalar = all(np.ndim(value) == 0 for value in (x, y, bearing, distance))
        angle = np.deg2rad(np.asarray(bearing, dtype=float))
        distance_array = np.asarray(distance, dtype=float)
        target_x = np.asarray(x, dtype=float) + distance_array * np.sin(angle)
        target_y = np.asarray(y, dtype=float) + distance_array * np.cos(angle)
        reverse = np.remainder(np.asarray(bearing, dtype=float) + 180.0, 360.0)
        return tuple(
            self._restore_scalar(np.asarray(value), scalar)
            for value in (target_x, target_y, reverse)
        )


def make_spatial_geometry(config: SpatialGeometryConfig) -> SpatialGeometry:
    if config.coordinate_system == "geographic":
        if config.ellipsoid is None:
            raise ValueError("geographic geometry requires an ellipsoid")
        return GeographicGeometry(config.ellipsoid, config.length_unit)
    if config.coordinate_system == "cartesian":
        return CartesianGeometry(config.length_unit)
    raise ValueError(f"unsupported coordinate system: {config.coordinate_system}")


def signed_angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return ``first - second`` wrapped to [-180, 180) degrees."""
    return (
        np.remainder(
            np.asarray(first, dtype=float) - np.asarray(second, dtype=float) + 180.0,
            360.0,
        )
        - 180.0
    )


def grid_array(cells: pd.DataFrame, grid: Any, field: str) -> np.ndarray:
    """Place a cell-table field on its configured two-dimensional grid."""
    values = np.full((grid.ny, grid.nx), np.nan, dtype=float)
    valid = cells.x_bin.between(0, grid.nx - 1) & cells.y_bin.between(
        0, grid.ny - 1
    )
    rows = cells.loc[valid]
    values[rows.y_bin.to_numpy(np.int64), rows.x_bin.to_numpy(np.int64)] = rows[
        field
    ].to_numpy(float)
    return values


def support_aware_uniform_3x3(
    values: np.ndarray,
    support: np.ndarray,
    *,
    periodic_x: bool,
) -> np.ndarray:
    """Mild uniform smoothing that never fills an unsupported focal cell."""
    values = np.asarray(values, dtype=float)
    support = np.asarray(support, dtype=bool) & np.isfinite(values)
    numerator = np.zeros_like(values, dtype=float)
    denominator = np.zeros_like(values, dtype=float)
    nlat, nlon = values.shape
    for delta_lat in (-1, 0, 1):
        source_lat_start = max(0, -delta_lat)
        source_lat_stop = min(nlat, nlat - delta_lat)
        target_lat_start = source_lat_start + delta_lat
        target_lat_stop = source_lat_stop + delta_lat
        source_values = values[source_lat_start:source_lat_stop]
        source_support = support[source_lat_start:source_lat_stop]
        for delta_lon in (-1, 0, 1):
            if periodic_x:
                shifted_values = np.roll(source_values, delta_lon, axis=1)
                shifted_support = np.roll(source_support, delta_lon, axis=1)
                numerator[target_lat_start:target_lat_stop] += np.where(
                    shifted_support, shifted_values, 0.0
                )
                denominator[target_lat_start:target_lat_stop] += shifted_support
            else:
                source_lon_start = max(0, -delta_lon)
                source_lon_stop = min(nlon, nlon - delta_lon)
                target_lon_start = source_lon_start + delta_lon
                target_lon_stop = source_lon_stop + delta_lon
                selected_values = source_values[:, source_lon_start:source_lon_stop]
                selected_support = source_support[:, source_lon_start:source_lon_stop]
                numerator[
                    target_lat_start:target_lat_stop,
                    target_lon_start:target_lon_stop,
                ] += np.where(selected_support, selected_values, 0.0)
                denominator[
                    target_lat_start:target_lat_stop,
                    target_lon_start:target_lon_stop,
                ] += selected_support
    smoothed = np.full_like(values, np.nan, dtype=float)
    valid = support & (denominator > 0)
    smoothed[valid] = numerator[valid] / denominator[valid]
    return smoothed


def bilinear_supported_sample(
    values: np.ndarray,
    support: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    grid: Any,
    *,
    weight_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate only when every nonzero-weight corner is supported."""
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)
    sampled = np.full(target_x.shape, np.nan, dtype=float)
    boundary = np.zeros(target_x.shape, dtype=bool)
    missing = np.zeros(target_x.shape, dtype=bool)
    span = grid.x_max - grid.x_min
    for index, (x_coordinate, y_coordinate) in enumerate(zip(target_x, target_y)):
        if not np.isfinite(x_coordinate) or not np.isfinite(y_coordinate):
            missing[index] = True
            continue
        if grid.periodic_x:
            x_coordinate = ((x_coordinate - grid.x_min) % span) + grid.x_min
        x = (x_coordinate - grid.x_min) / grid.dx - 0.5
        y = (y_coordinate - grid.y_min) / grid.dy - 0.5
        if y < -weight_tolerance or y > grid.ny - 1 + weight_tolerance:
            boundary[index] = True
            continue
        if not grid.periodic_x and (
            x < -weight_tolerance or x > grid.nx - 1 + weight_tolerance
        ):
            boundary[index] = True
            continue
        if not grid.periodic_x:
            x = min(max(x, 0.0), grid.nx - 1.0)
        y = min(max(y, 0.0), grid.ny - 1.0)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        candidates = (
            (y0, x0, (1.0 - fx) * (1.0 - fy)),
            (y0, x0 + 1, fx * (1.0 - fy)),
            (y0 + 1, x0, (1.0 - fx) * fy),
            (y0 + 1, x0 + 1, fx * fy),
        )
        weighted_value = 0.0
        defensible = True
        for lat_index, lon_index, weight in candidates:
            if weight <= weight_tolerance:
                continue
            if lat_index < 0 or lat_index >= grid.ny:
                boundary[index] = True
                defensible = False
                break
            if grid.periodic_x:
                lon_index %= grid.nx
            elif lon_index < 0 or lon_index >= grid.nx:
                boundary[index] = True
                defensible = False
                break
            if not support[lat_index, lon_index] or not np.isfinite(
                values[lat_index, lon_index]
            ):
                missing[index] = True
                defensible = False
                break
            weighted_value += weight * values[lat_index, lon_index]
        if defensible:
            sampled[index] = weighted_value
    return sampled, boundary, missing


def physical_cell_scales(
    cells: pd.DataFrame,
    grid: Any,
    geometry: SpatialGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, and geometric-mean cell scales in configured length units."""
    x = cells.x.to_numpy(float)
    y = cells.y.to_numpy(float)
    _, _, x_scale = geometry.inverse(
        x - grid.dx / 2.0, y, x + grid.dx / 2.0, y
    )
    _, _, y_scale = geometry.inverse(
        x, y - grid.dy / 2.0, x, y + grid.dy / 2.0
    )
    effective = np.sqrt(np.asarray(x_scale) * np.asarray(y_scale))
    return np.asarray(x_scale), np.asarray(y_scale), effective


# Private aliases keep the verified Stage-5/6/7 implementations numerically intact.
_signed_difference = signed_angle_difference
_grid_array = grid_array
_bilinear_supported_sample = bilinear_supported_sample
_physical_cell_scales = physical_cell_scales
