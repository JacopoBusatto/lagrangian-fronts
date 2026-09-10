# Changelog

## 0.1.0 — First standalone version

- Extracted the validated trajectory → transition matrix → flux/directional front workflow into an independent `src/lagrangian_fronts` package.
- Added local installation, the `lagrangian-fronts` command, and `python -m lagrangian_fronts`, each using the existing single YAML configuration.
- Preserved configuration version 1, matrix schemas, scientific calculations, thresholds, ridge/plateau selection, front detection and output contracts.
- Migrated scientific tests and reference fixtures, portable examples, documentation, and the existing MIT license.

This version packages the existing scientific method. It has not been published to PyPI and has no remote repository URL or paper DOI yet.
