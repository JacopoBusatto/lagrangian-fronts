# Trajectory input contract

Trajectories supplied to this module must already be scientifically prepared. No ARGO quality flags, drogue/depth selection, speed-jump filters, repair, or other platform-specific QC is applied.

## Canonical fields

| Field | Meaning |
| --- | --- |
| `trajectory` | Nonmissing scalar identity; file scope is the default. |
| `time` | Internally, floating-point elapsed seconds from the first retained observation of each trajectory/group member. |
| `x`, `y` | Finite longitude/latitude in degrees, or planar coordinates in `geometry.length_unit`. |
| `obs` | Optional deterministic secondary key for observations at the same time. |
| `group_member` | Optional identity component; members are never joined into one track. |

Readers return `TrajectoryData(table, diagnostics, input_files)`. The matrix builder also accepts an already-canonical dataframe: it must have finite numeric time/x/y, nonmissing identities, and at least two strictly time-ordered observations per track. Use `read_trajectories` to perform sanitation automatically. The canonical dataframe is not saved during normal execution.

## Geographic example

```csv
float_id,time,longitude,latitude
5900001,2026-01-01T00:00:00,20.2,-48.4
5900001,2026-01-11T00:00:00,21.0,-48.2
5900002,2026-01-01T00:00:00,18.7,-50.1
5900002,2026-01-11T00:00:00,19.4,-49.9
```

```yaml
input:
  paths: [data/floats.csv]
  format: auto
  columns: {trajectory: float_id, time: time, x: longitude, y: latitude}
  trajectory_id: {mode: column, scope: file}
  time: {kind: datetime}
```

Datetime strings and decoded datetime arrays are accepted. Time-zone-aware timestamps are converted to UTC; naive timestamps are interpreted as UTC. Numeric values must not be presented as datetime unless a NetCDF/Zarr CF time encoding decodes them into real timestamps. Standard/Gregorian calendars are supported; convert nonstandard calendars to numeric elapsed time upstream.

Geographic latitude must be in `[-90, 90]`. Equivalent longitudes are mapped onto the 360-degree branch starting at `grid.lon_min`; regional bounds still reject points outside the configured interval. Longitude interpolation takes the shortest angular arc, retaining the existing +/-180-degree tie convention. Positions exactly on a nonperiodic maximum boundary are outside. At a global longitude seam the upper meridian maps to the lower meridian.

## Cartesian example

```csv
particle,t,xpos,ypos
0,0,1.0,2.0
0,2,1.2,2.0
1,0,4.0,3.0
1,2,4.1,3.1
```

```yaml
input:
  paths: [data/particles.csv]
  format: csv
  columns: {trajectory: particle, time: t, x: xpos, y: ypos}
  time: {kind: numeric, unit: s}
```

Numeric time supports `s`, `min`, `h`, and `day`. It may start at any finite origin. Its unit describes the input samples; `matrix.time_unit` separately describes the requested transition lag and the units of flux-rate outputs. Cartesian coordinates are nonperiodic and require no CRS.

## Multiple files and identity

```yaml
input:
  paths: [experiment_A/*.parquet, experiment_B/*.parquet]
```

Paths and globs resolve against the YAML directory. Every pattern must match. Resolved paths are deduplicated and sorted before reading, so overlapping globs do not double-count a file. `auto` detects `.csv`, `.parquet`/`.pq`, `.nc`/`.nc4`/`.netcdf`, and directory stores ending in `.zarr`. An explicit format supports otherwise unnamed extensions. Remote URLs, zipped Zarr stores and arbitrary custom formats are not supported.

In `column` mode, `columns.trajectory` is required. IDs become strings, with missing values remaining missing. CSV identifiers are read as strings to preserve leading zeros. With default `scope: file`, canonical IDs encode `[relative_source_path, source_id]` as JSON strings. Relative paths use `/`; inputs on another Windows drive use their absolute path. With `scope: global`, matching source IDs and group members join across files before time normalization. Use global scope only when IDs refer to the same physical track across all inputs. Global numeric time must use a shared origin across those fragments.

In `filename` mode, one file means one trajectory. A source ID column is unnecessary, and canonical identity is the JSON encoding of `[relative_source_path]`. Identical basenames in different directories remain distinct. `scope` has no additional effect in this mode. `resolved_input_base` in generated YAML preserves the original path basis when replaying a saved configuration.

## NetCDF and Zarr layouts

Configured names may identify variables, coordinates, or bare dimensions. For example, these arrays form a supported rectangular dataset:

```text
particle(particle)              # coordinate, variable, or dimension index
observation(observation)
t(observation)                 # shared sampling times
X(particle, observation)
Y(particle, observation)
```

Map `trajectory: particle`, `time: t`, `x: X`, `y: Y`, `obs: observation`. Per-particle time arrays and long observation tables are also supported. One x/y dimension set must contain the other; all mapped fields must fit that observation shape. Scalar or lower-dimensional fields broadcast only onto that shape. Unrelated extra dimensions are rejected instead of introducing combinations of unrelated observations. Bare dimensions use deterministic zero-based indices.

Only mapped fields are read and flattened. All explicitly mapped fields must exist, including optional fields mapped to non-null names. Padded missing samples are removed during sanitation. Specialized indexed/contiguous ragged encodings, unusual string layouts and nonstandard calendars must be converted upstream. A new format can be added through `READERS`; matrix and scientific functions need no changes.

## Structural sanitation and diagnostics

Sanitation removes missing identities, invalid time, nonfinite coordinates, and invalid geographic latitude. Observations sort by identity, time, then `obs` if mapped. Sorting is stable: remaining ties follow sorted file order and source row/array order. The first duplicate time is kept. Tracks with fewer than two remaining samples are discarded.

`invalid_observations` and `duplicate_time_observations` are disjoint; their sum is `structurally_rejected_observations`. `observations_in_short_trajectories` counts additional otherwise-valid samples excluded with short tracks. `valid_observations` is the final retained count. Tracks with identifiable IDs but no valid observations are reported separately as `discarded_empty_trajectories`. `source_trajectories` counts identifiable trajectory/group-member keys before sanitation.

All adapters materialize selected observations in memory in this version. Sorting and concatenation need additional working memory. Sparse transition counts are accumulated per trajectory; a dense cell-to-cell matrix is never allocated. Chunked/distributed reading and QC are deferred.
