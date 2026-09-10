# Scientific reference fixtures

`cartesian/` and `geographic/` were captured from commit
`10e63b4675a6b6d3f9eb1746bc9ecb6b04bf0ed5` before the migration. Each contains an
11×11 synthetic stay/move probability ridge, its configuration, all dataframe
fields in the pipeline result objects, and the remaining scalar/summary fields.
Both cases contain detected transport and directional cores and fronts.

Configurations were mechanically migrated to schema version 1. Expected numeric
outputs were not regenerated from the new implementation. `test_regression.py`
compares every dataframe and summary exactly at the same Parquet serialization
boundary. This preserves numerical values while accounting for Parquet's bool
dtype inference for baseline Python-object columns.

Full ARGO/drifter baseline data and comparisons remain private historical
validation artifacts and are not distributed with this package. The compact
fixtures run without those datasets, Git, or a kinematicparcels installation.

The flux terminology migration leaves every captured configuration, Parquet
table and metadata file unchanged. Tests translate the captured configuration
into a temporary file and compare results using the explicit name, category and
quality-flag mappings in `test_regression.py`. Only names are translated;
numerical values, dtypes, row/column ordering and scientific selections retain
their exact assertions. Historical `transport` filenames remain part of these
reference captures.
