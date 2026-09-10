# Configuration reference

One YAML controls the complete workflow. `lagrangian-fronts --config analysis.yaml` and `python -m lagrangian_fronts --config analysis.yaml` are equivalent. Configuration version 1 is independent of software version 0.1.0. It rejects unknown keys, unsupported versions, wrong scalar types, nonfinite numbers and invalid ranges. The examples use deliberate synthetic settings; dataclass defaults below and current Southern Ocean settings are preserved separately.

Required top-level sections are `config_version`, `matrix`, `geometry`, `grid`, and `output`. `input` is required only when computing a matrix. Other sections are optional and fill in the defaults below. `geometry` and `grid` must describe the same coordinate system. Relative paths resolve from the YAML file, including the output root. No environment-variable substitution or CLI parameter overrides are performed.

The public structure is:

```yaml
config_version: 1
input:  # compute mode only
  paths: [data/tracks.csv]
  columns: {trajectory: particle, time: t, x: xpos, y: ypos}
  time: {kind: numeric, unit: s}
matrix: {compute: true, timestep: 2, time_unit: s}
geometry: {coordinate_system: cartesian, length_unit: m}
grid: {x_min: 0, x_max: 11, y_min: 0, y_max: 11, dx: 1, dy: 1}
statistics: {}
flux: {}
directional: {}
fronts: {}
validation: {}
output: {root: outputs, run_name: example}
plotting: {enabled: false}
write_debug_outputs: false
run_validation: false
```

For load mode use `matrix.compute: false`, provide `matrix.path`, and keep explicit positive timestep/time_unit. Input is optional and is not opened. A recognized `matrix_summary.yaml` must match the matrix SHA-256, geometry, grid and physical lag; probabilities are never repaired. The stored summary retains the original segmentation/gap metadata when available.

## Top-level switches

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `config_version` | integer | required; no YAML default | 1 only | none | Public YAML schema version. Unsupported or missing versions are rejected; version 1 does not silently accept legacy matrix-input YAML. |
| `analysis_version` | string | optional; `5.0.0-production` | string | none | Recorded scientific analysis label. Changing this metadata does not change calculations or the pinned baseline commit. |
| `write_debug_outputs` | boolean | optional; `false` | true/false | none | Write additional flux/directional component, graph, raw section, composite and candidate-drop tables under analysis/. Increases disk use; does not change selections. |
| `run_validation` | boolean | optional; `false` | true/false | none | Calculate the independent flux-gradient comparison and save gradient_validation.parquet. It never changes core/front selection. |

## Input

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `input.paths` | list of strings | required (when input configured) | nonempty list of nonblank strings | none / dimensionless | Input files or glob patterns. Resolve against the YAML directory, sort and deduplicate matches. Every pattern must match; file scope keeps experiments separate. |
| `input.columns` | mapping | required in compute mode | mapping | none / dimensionless | Nested mapping; parameters listed below. |
| `input.time` | mapping | required in compute mode | mapping | see kind/unit | Nested mapping; parameters listed below. |
| `input.format` | string | optional; `auto` (when input configured) | auto, csv, parquet, netcdf, zarr | none / dimensionless | Reader selection. auto recognizes documented extensions; explicit formats allow other filenames. Zarr requires a local directory store. |
| `input.trajectory_id` | mapping | optional; defaults below | mapping | none / dimensionless | Nested mapping; parameters listed below. |
| `input.columns.time` | string | required (when input configured) | nonblank source name or null | none / dimensionless | Source observation time field; decoding is governed by input.time. It may be an array or a shared coordinate. |
| `input.columns.x` | string | required (when input configured) | nonblank source name or null | none / dimensionless | Source longitude or Cartesian x field. Renamed to canonical x before matrix construction; no CRS conversion is performed. |
| `input.columns.y` | string | required (when input configured) | nonblank source name or null | none / dimensionless | Source latitude or Cartesian y field. Renamed to canonical y before matrix construction. |
| `input.columns.trajectory` | string or null | required in column mode; null otherwise | nonblank source name or null | none / dimensionless | Source identity column, variable, coordinate or dimension. Required in column mode; ignored in filename mode. |
| `input.columns.obs` | string or null | optional; `null` (when input configured) | nonblank source name or null | none / dimensionless | Optional source observation-order field. Breaks ties at duplicate times before keeping the first observation; changing this can change the retained position. |
| `input.columns.group_member` | string or null | optional; `null` (when input configured) | nonblank source name or null | none / dimensionless | Optional additional identity field. Observations from different group members cannot form a shared transition. |
| `input.trajectory_id.mode` | string | optional; `column` (when input configured) | column, filename | none / dimensionless | column reads mapped IDs; filename explicitly declares one trajectory per file and derives its identity from its path. |
| `input.trajectory_id.scope` | string | optional; `file` (when input configured) | file, global | none / dimensionless | In column mode, file namespaces IDs by source path; global joins matching IDs and group members across files before sorting. No additional effect in filename mode. |
| `input.time.kind` | string | required (when input configured) | datetime, numeric | see kind/unit | datetime parses timestamps (UTC); numeric uses simulation/elapsed values. Both become per-track elapsed seconds before the matrix kernel. |
| `input.time.unit` | string or null | required for numeric; null otherwise | s, min, h, day; null for datetime | see kind/unit | Scale of numeric source time. Required for numeric input; must be null/omitted for datetime. Independent of matrix.time_unit. |

## Matrix

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `matrix.timestep` | number | required | finite > 0; elapsed seconds must remain finite | matrix.time_unit | Elapsed time represented by one transition. Changes the temporal/spatial connectivity scale and the rate denominator. Any positive lag is allowed; inferred source sampling does not restrict it. |
| `matrix.time_unit` | string | required | s, min, h, day | none / dimensionless | Unit of timestep and max_interpolation_gap, and the time denominator used in reported flux-rate units. Its physical elapsed duration must agree with a recognized matrix sidecar. |
| `matrix.compute` | boolean | optional; `true` | true/false | none / dimensionless | true reads trajectories, computes and saves a matrix; false reads matrix.path. Both use exactly the same downstream scientific API. |
| `matrix.path` | string or null | optional; `null` | string | none / dimensionless | Existing Parquet matrix. Required in load mode and null in compute mode; relative paths resolve against YAML. The matrix is validated, never silently repaired. |
| `matrix.id` | string or null | optional; `null` | nonblank string or null | none / dimensionless | Recorded matrix label, defaulting to output.run_name when null. Does not affect transition counts or scientific selection. |
| `matrix.segment_mode` | string | optional; `non_overlapping` | non_overlapping, every_observation | none / dimensionless | non_overlapping anchors consecutive lag-sized intervals at the first valid time; every_observation starts at every observation except the last. The latter samples overlapping intervals and changes statistical weighting. Ignored when loading a matrix; a sidecar preserves the original construction metadata. |
| `matrix.max_interpolation_gap` | number or null | optional; `null` | null or finite > 0 | matrix.time_unit | Reject a boundary requiring interpolation when its bracketing observation interval exceeds this maximum. Null disables the safeguard. Exact boundaries remain valid; later non-overlapping anchors do not shift. Used only in compute mode. |

## Geometry and grid

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `geometry.coordinate_system` | string | required | geographic, cartesian | none / dimensionless | geographic uses degree coordinates and ellipsoidal geodesics; cartesian uses Euclidean x/y distances. Selects grid and public matrix column names. |
| `geometry.length_unit` | string | required | mm, cm, m, km | none / dimensionless | Physical length unit for displacements and output distances. Cartesian input coordinates and grid spacing already use this unit. Geographic distances are converted from metres. Changing units requires consistent Cartesian inputs/grid. |
| `geometry.ellipsoid` | string or null | required geographically; null otherwise | valid pyproj ellipsoid name; null in Cartesian mode | none / dimensionless | pyproj.Geod ellipsoid name for geographic distances/bearings. Required geographically and forbidden in Cartesian mode. Changing it changes physical geometry. |
| `grid.lon_min` | number | required | finite; positive spacing; integer cell counts; maxima > minima | degrees | Explicit longitude grid lower (included) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.lon_max` | number | required | finite; positive spacing; integer cell counts; maxima > minima | degrees | Explicit longitude grid upper (excluded) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.lat_min` | number | required | finite; positive spacing; integer cell counts; maxima > minima; latitude in [-90,90] | degrees | Explicit latitude grid lower (included) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.lat_max` | number | required | finite; positive spacing; integer cell counts; maxima > minima; latitude in [-90,90] | degrees | Explicit latitude grid upper (excluded) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.dlon` | number | required | finite; positive spacing; integer cell counts; maxima > minima | degrees | Regular longitude cell spacing. Must divide its configured range into an integer number of cells. Changes cell-centre geometry, stay/moving separation and support. |
| `grid.dlat` | number | required | finite; positive spacing; integer cell counts; maxima > minima | degrees | Regular latitude cell spacing. Must divide its configured range into an integer number of cells. Changes cell-centre geometry, stay/moving separation and support. |
| `grid.periodic_longitude` | boolean | optional; `true` | true/false | none | Wrap scientific grid neighborhoods across the longitudinal grid seam. Does not wrap latitude. Global geographic grids normally enable it; regional examples disable it. |
| `grid.x_min` | number | required | finite; positive spacing; integer cell counts; maxima > minima | geometry.length_unit | Explicit planar x grid lower (included) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.x_max` | number | required | finite; positive spacing; integer cell counts; maxima > minima | geometry.length_unit | Explicit planar x grid upper (excluded) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.y_min` | number | required | finite; positive spacing; integer cell counts; maxima > minima | geometry.length_unit | Explicit planar y grid lower (included) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.y_max` | number | required | finite; positive spacing; integer cell counts; maxima > minima | geometry.length_unit | Explicit planar y grid upper (excluded) bound. Changing bounds changes the retained transition population and its normalization. |
| `grid.dx` | number | required | finite; positive spacing; integer cell counts; maxima > minima | geometry.length_unit | Regular x cell spacing. Must divide its configured range into an integer number of cells. Changes cell-centre geometry, stay/moving separation and support. |
| `grid.dy` | number | required | finite; positive spacing; integer cell counts; maxima > minima | geometry.length_unit | Regular y cell spacing. Must divide its configured range into an integer number of cells. Changes cell-centre geometry, stay/moving separation and support. |

## Scientific statistics and selection

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `statistics.min_moving_support` | integer | optional; `10` | integer > 0 | none / dimensionless | Minimum moving transition count `N_out_move` for all core, corridor, front, comparison, and plotted selections. Lower values retain more weakly sampled cells; higher values are more conservative. |
| `statistics.angular_bins` | integer | optional; `36` | integer >= 4 | none / dimensionless | Number of equal bearing bins used for outgoing and incoming angular entropy; must be at least four. It does not change the first harmonic `R1`. |
| `statistics.direction_zero_tolerance` | number | optional; `1e-12` | finite >= 0 | length_unit for displacement; dimensionless for harmonics | Numerical tolerance, in the configured length unit for displacement vectors, below which magnitude is treated as zero and its bearing is undefined. |
| `statistics.high_R1` | number | optional; `0.8` | low_R1 <= value <= 1 | none / dimensionless | High first-harmonic threshold used in statistical diagnostic categories and summaries. It is not the corridor-selection threshold. |
| `statistics.low_R1` | number | optional; `0.5` | 0 <= value <= high_R1 | none / dimensionless | Low first-harmonic threshold used in diagnostic categories; it must not exceed `high_R1`. |
| `flux.percentile` | number | optional; `0.9` | 0 < value < 1 | none / dimensionless | Quantile of the selected flux field above which transverse ridge candidates are retained. Lowering it yields more cores; increasing it retains only stronger flux. Must lie strictly between zero and one. |
| `flux.ridge_field` | string | optional; `raw` | raw, smoothed | none / dimensionless | Core field: `raw` uses `|U_out_all|`; `smoothed` uses a support-aware 3 x 3 mean. |
| `flux.transverse_scale_grid` | number | optional; `1.0` | finite > 0 | local effective grid scale | Cross-stream distance for sampling the flux field on either side of a candidate. Increasing it compares the centre against a broader structure and can change ridge membership. |
| `flux.interpolation_weight_tolerance` | number | optional; `1e-10` | finite >= 0 | none / dimensionless | Numerical bilinear-weight cutoff used only by the flux pathway. Corners with weights at or below this value are ignored; every other contributing corner must be supported. |
| `flux.orientation_reliable_R1` | number | optional; `0.8` | orientation_ambiguous_R1 <= value <= 1 | none / dimensionless | `R1_out` threshold for the reliable-orientation diagnostic. It does not filter core membership. |
| `flux.orientation_ambiguous_R1` | number | optional; `0.5` | 0 <= value <= orientation_reliable_R1 | none / dimensionless | `R1_out` threshold for the ambiguous-orientation diagnostic. It does not filter core membership. |
| `flux.direction_disagreement_degrees` | number | optional; `20.0` | 0 <= value <= 180 | degrees | Maximum `theta_mu_out` versus `theta1_out` disagreement used by the reliable-orientation diagnostic; not a core filter. |
| `flux.abrupt_tangent_mismatch_degrees` | number | optional; `45.0` | 0 <= value <= 180 | degrees | Threshold used to flag a large mismatch between a core graph segment and the mean flux direction; diagnostic only. |
| `flux.ridge_comparison_tolerance` | number | optional; `1e-12` | finite >= 0 | length_unit / matrix.time_unit | Numerical allowance in the configured rate unit for the centre-versus-flank ridge comparison. |
| `flux.smoothing_window_cells` | integer | optional; `3` | 3 only | samples/sections/cells as named | Fixed validated smoothing window. The current implementation accepts only `3`. |
| `directional.minimum_P_move` | number | optional; `0.5` | 0 <= value <= 1 | none / dimensionless | Minimum probability that a transition leaves its source cell. |
| `directional.minimum_R1` | number | optional; `0.8` | 0 <= value <= 1 | none / dimensionless | Minimum concentration of moving-transition bearings. |
| `directional.minimum_strength` | number | optional; `0.5` | 0 <= value <= 1 | none / dimensionless | Minimum dimensionless `P_move * R1_out`. |
| `directional.maximum_neighbor_direction_difference_degrees` | number | optional; `45.0` | 0 <= value <= 90 | degrees | Largest circular `theta1_out` difference allowed between connected candidate cells. |
| `directional.maximum_step_direction_mismatch_degrees` | number | optional; `45.0` | 0 <= value <= 90 | degrees | Largest mismatch allowed between the geodesic cell-to-cell axis and the local directional axis at either endpoint. |
| `directional.minimum_component_cells` | integer | optional; `3` | integer >= 1 | samples/sections/cells as named | Minimum number of connected ridge cells required to retain a final directional-core component. Candidate components are not size-filtered. |
| `directional.transverse_scale_grid` | number | optional; `1.0` | finite > 0 | local effective grid scale | Cross-stream distance at which D = P_move × R1 is sampled on either side of the candidate centre. Increasing it compares the centre to a broader transverse structure and can change ridge membership. |
| `directional.ridge_comparison_tolerance` | number | optional; `1e-12` | finite >= 0 | none / dimensionless | Dimensionless numerical allowance in the centre-versus-flank `D` comparison. |
| `directional.ridge_plateau_tolerance` | number | optional; `1e-12` | finite >= 0 | none / dimensionless | Maximum pairwise D difference for adjacent raw ridge cells satisfying the frozen cross-stream step criterion to join a plateau. Connectivity is transitive. A larger tolerance can merge more ridges before medoid thinning; this is not a global directional percentile. |
| `directional.interpolation_weight_tolerance` | number | optional; `1e-10` | finite >= 0 | none / dimensionless | Directional-only bilinear-weight cutoff used by ridge and directional-front sampling. |

## Fronts and validation

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `fronts.half_width_grid_scales` | integer | optional; `5` | integer >= 2 | local effective grid scale | Distance sampled on each side of a flux or directional-core axis; must be at least two grid scales. |
| `fronts.sampling_interval_grid_scales` | number | optional; `1.0` | finite > 0 | local effective grid scale | Spacing between transverse samples. Smaller values give denser interpolation and higher cost. |
| `fronts.core_refinement_grid_scales` | number | optional; `1.0` | 0 < value <= 1 | local effective grid scale | Maximum distance over which the original core axis is shifted to the local projected-field maximum; must be in `(0, 1]`. |
| `fronts.robust_median_window_samples` | integer | optional; `3` | positive odd integer | samples/sections/cells as named | Positive odd window for the contiguous rolling-median profile. Missing data split the profile into separate runs. |
| `fronts.composite_half_window_sections` | integer | optional; `2` | integer >= 0 | samples/sections/cells as named | Context used for composite profiles: plus/minus this many ordered sections on a flux segment, or this many graph hops for a directional core. Zero uses only the focal section. |
| `fronts.minimum_persistent_neighbor_sections` | integer | optional; `2` | integer >= 1 | samples/sections/cells as named | Minimum number of composite sections that must show an outward decline when neighbour information is available. |
| `fronts.minimum_persistent_fraction` | number | optional; `0.5` | 0 <= value <= 1 | none / dimensionless | Minimum fraction of available composite sections that must show the decline. Larger values make front selection stricter. |
| `fronts.diagnostic_low_R1` | number | optional; `0.5` | finite number | none / dimensionless | Flags low directional concentration on flux-front sections; diagnostic only. |
| `fronts.diagnostic_large_direction_disagreement_degrees` | number | optional; `20.0` | finite number | degrees | Flags disagreement between mean-vector and first-harmonic bearings on flux-front sections; diagnostic only. |
| `fronts.diagnostic_high_curvature_degrees` | number | optional; `60.0` | finite number | degrees | Flags highly turning flux-core segments; diagnostic only. |
| `fronts.diagnostic_strong_outer_recovery_fraction` | number | optional; `0.5` | finite number | none / dimensionless | Flags a flux profile whose signal recovers outward by more than this fraction of its selected drop; it does not reject the front. |
| `fronts.diagnostic_min_full_section_valid_samples` | integer | optional; `7` | integer | samples/sections/cells as named | Minimum valid samples before a section avoids the `short_available_cross_section` quality flag. |
| `fronts.nearby_branch_cross_distance_scales` | number | optional; `5.0` | finite number | none / dimensionless | Cross-stream search distance used to flag contamination by another flux-core component; diagnostic only. |
| `fronts.nearby_branch_along_distance_scales` | number | optional; `1.0` | finite number | none / dimensionless | Along-stream search distance used with the preceding contamination diagnostic. |
| `validation.normalization_atol` | number | optional; `1e-12` | finite >= 0 | none / dimensionless | Absolute tolerance for requiring each populated source row to sum to probability one. |
| `validation.probability_rtol` | number | optional; `1e-10` | finite >= 0 | none / dimensionless | Relative tolerance for checking `transition_probability = transition_count / N_out_total`. |
| `validation.probability_atol` | number | optional; `1e-12` | finite >= 0 | none / dimensionless | Absolute tolerance for the same count/probability identity. |
| `validation.center_atol` | number | optional; `1e-09` | finite >= 0 | degrees or length_unit | Absolute centre-coordinate tolerance: degrees geographically or `geometry.length_unit` in Cartesian mode. |
| `validation.gradient_zero_tolerance` | number | optional; `1e-12` | finite >= 0 | 1 / matrix.time_unit | Gradient magnitude at or below which the optional gradient orientation is considered undefined. |
| `validation.interpolation_weight_tolerance` | number | optional; `1e-10` | finite >= 0 | none / dimensionless | Bilinear-weight cutoff used only when sampling optional validation gradients. |
| `validation.gradient_search_radius_grid_scales` | number | optional; `1.0` | finite >= 0 | local effective grid scale | Radius around a selected flux front used to find the local maximum transverse gradient. |
| `validation.local_background_radius_grid_scales` | number | optional; `2.0` | finite >= 0 | local effective grid scale | Radius around the refined core used to calculate the front gradient's local percentile. |
| `validation.duplicate_disagreement_grid_scales` | number | optional; `1.0` | finite number | local effective grid scale | Candidate-position spread above which duplicate segment-context fronts for one core side are flagged as disagreeing. |
| `validation.core_gradient_ratio_epsilon` | number | optional; `1e-12` | finite number | 1 / matrix.time_unit | Denominator guard for the optional flank-to-core gradient ratio. |
| `validation.multiple_drop_similarity_fraction` | number | optional; `0.1` | finite number | none / dimensionless | Reserved configuration field; it is currently not used by the production or validation calculations. |
| `validation.direct_sample_atol_grid_cells` | number | optional; `1e-08` | finite >= 0 | samples/sections/cells as named | Grid-coordinate tolerance for labeling a validation gradient sample as direct rather than interpolated. |

## Output and figures

| YAML path | Type | Required / default | Allowed values | Units | Meaning and effect |
| --- | --- | --- | --- | --- | --- |
| `output.root` | string | required | nonblank string | none / dimensionless | Parent output directory, relative to YAML when not absolute. No automatic timestamp is added. |
| `output.run_name` | string | optional; `lagrangian_currents` | nonblank single directory name; not . or .. | none / dimensionless | Exact run-directory name beneath output.root; the destination must not already exist. A new name permits another run with identical science. |
| `plotting.enabled` | boolean | optional; `true` | true/false | none / dimensionless | Create the standard figures. |
| `plotting.dpi` | integer | optional; `160` | integer > 0 | pixels per inch | Positive raster resolution used when saving figures. |
| `plotting.projection` | string | optional; `SouthPolarStereo` | PlateCarree, SouthPolarStereo | none / dimensionless | Geographic map projection; supported values are `SouthPolarStereo` and `PlateCarree`. Ignored in Cartesian mode. |
| `plotting.central_longitude` | number | optional; `0.0` | finite number | degrees | Central longitude of the polar stereographic projection; geographic only. |
| `plotting.circular_boundary` | boolean | optional; `true` | true/false | none / dimensionless | Clip South Polar Stereographic axes to a circular boundary; geographic only. |
| `plotting.draw_coastlines` | boolean | optional; `true` | true/false | none / dimensionless | Draw Cartopy coastlines; geographic only. |
| `plotting.vector_stride_cells` | integer | optional; `5` | integer >= 1 | samples/sections/cells as named | Plot one flux/directional arrow every this many grid cells in both dimensions. |
| `plotting.vector_reference` | number | optional; `5.0` | finite > 0 (null is not automatic scaling) | length_unit / matrix.time_unit | Reference-arrow magnitude in the configured length/time rate unit. |
| `plotting.directional_vector_reference` | number | optional; `0.5` | 0 < value <= 1 | none / dimensionless | Dimensionless reference-arrow magnitude for directional-vector maps; must be in `(0, 1]`. |
| `plotting.structure_map_max_percentile` | number | optional; `100.0` | 0 < value <= 100 | none / dimensionless | Percentile used only for the upper colour limit in the structure and optional validation maps. |
| `plotting.debug_plots` | boolean | optional; `false` | boolean; true requires run_validation=true | none / dimensionless | Create the optional gradient-validation map. Requires `run_validation: true`. |

All numeric diagnostic settings must be finite. Some inherited diagnostic-only fields accept any finite value; their scientifically usual interpretation is explained in the table rather than silently imposing a new selection rule. The reserved `validation.multiple_drop_similarity_fraction` has no computational effect. The flux/displacement and directional tolerances have different units: do not interchange them.

## Generated metadata and migration

`resolved_config.yaml` expands defaults and records resolved paths. Its optional generated `resolved_geometry` mapping contains coordinate system/unit, length unit, time/rate units, bearing convention and geometry backend; the loader checks it against the configured geometry. Its optional `resolved_input_base` string records the original absolute path basis for filename-derived identities. These are replay metadata, not scientific overrides. The manifest separately records the original config path and source file fingerprints.

To migrate pre-versioned research YAML: add `config_version: 1`; move `input.transition_table` to `matrix.path`, `input.matrix_id` to `matrix.id`, and input timestep/time_unit to matrix; set `matrix.compute: false`; rename `branches.transport_percentile` to `flux.percentile`, the remaining branches keys to flux, and edges to fronts. Preserve numeric values. The portable Southern Ocean templates in `examples/production/` already apply these changes and retain the production thresholds; supply your own matrix paths. Current version-1 flux YAML requires no scientific changes for standalone installation.

### Flux terminology migration

The unreleased `config_version: 1` interface now uses **flux** for the Lagrangian displacement-flux pathway. Rename `transport:` to `flux:` without changing nested values. The old section is rejected with a migration error, including when both names appear. No YAML or Python compatibility aliases are provided.

| Earlier interface | Current interface |
| --- | --- |
| `TransportConfig`, `config.transport` | `FluxConfig`, `config.flux` |
| `AnalysisResult.transport_cores`, `.transport_fronts` | `.flux_cores`, `.flux_fronts` |
| `compare_transport_and_directional_structures` | `compare_flux_and_directional_structures` |
| `transport_cores.parquet`, `transport_fronts.parquet` | `flux_cores.parquet`, `flux_fronts.parquet` |
| `S_transport_rate`, `S_transport` | `S_flux_rate`, `S_flux` |
| `transport_loss_rate`, `absolute_transport_loss`, `relative_transport_loss` | `flux_loss_rate`, `absolute_flux_loss`, `relative_flux_loss` |
| `integrated_transport_area_rate` | `integrated_ridge_strength_area_rate` |
| `rank_integrated_transport`, `rank_median_transport` | `rank_integrated_ridge_strength`, `rank_median_flux` |
| `transport_component_id`, `transport_core` | `flux_component_id`, `flux_core` |
| `transport`, `transport_only`, `transport_and_directional` categories | `flux`, `flux_only`, `flux_and_directional` |
| `probable_transport_front`, `no_transport_core_sections`, `transport_undefined` statuses/flags | `probable_flux_front`, `no_flux_core_sections`, `flux_undefined` |
| `opposing_outer_transport` diagnostic/flag | `opposing_outer_flux` |
| `transport_threshold_*`, `transport_directional_comparison`, `probable_transport_fronts` manifest keys | `flux_threshold_*`, `flux_directional_comparison`, `probable_flux_fronts` |
| `01_transport_vectors.png`, `05_cores_and_fronts.png` | `01_flux_vectors.png`, `05_flux_cores_and_fronts.png` |
| `plot_transport_cores_hydrography.py` | `plot_flux_cores_hydrography.py` |

Related comparison fractions and validation summaries follow the same rename. Public unit suffixes retain their previous units: flux strengths and absolute losses use length/time, relative losses are dimensionless, and integrated ridge strength uses length²/time. `U_*`, matrices, formulas, thresholds and directional selections retain their definitions. Update downstream consumers of renamed fields and filenames. Existing run directories and captured reference artifacts retain their original names.

### Output inventory

The compact production tables are `cell_statistics.parquet`, `flux_cores.parquet`, `flux_fronts.parquet`, `directional_corridors.parquet` (candidate area with ridge/thinning diagnostics), `directional_cores.parquet`, `directional_fronts.parquet`, `comparison.parquet` and `component_comparison.parquet`, all under `analysis/`. Optional validation adds `gradient_validation.parquet`.

`write_debug_outputs` additionally writes flux component graph details, raw cross sections, section composites, candidate drop zones and segment-context front candidates; directional counterparts include candidate/core component and graph-edge tables. These retain the original scientific diagnostics without duplicating the compact production selections. `plotting.debug_plots` controls optional validation figures separately from debug tables; `plotting.enabled: false` disables every figure.

`matrix/matrix_summary.yaml` stores construction/validation counts, geometry/grid, physical lag, segmentation/gap settings where known, matrix path and hash. See [METHODOLOGY.md](METHODOLOGY.md) for counter reconciliation and [INPUT_DATA.md](INPUT_DATA.md) for sanitation counts. Only computed matrices are copied into the run folder. A loaded matrix remains at its original recorded location.
