# Standalone extraction report

## Reference and scope

The source was the validated working implementation at `kinematicParcels/research/lagrangian_fronts/`, including its completed flux terminology migration. The parent repository's HEAD was `10e63b4675a6b6d3f9eb1746bc9ecb6b04bf0ed5`, but the working module, not that earlier commit alone, was the extraction reference. Before changing anything, the relevant parent suite passed **231 tests, 1 warning in 35.66 s**:

```bash
python -m pytest tests/lagrangian_fronts tests/test_gridded_transition_matrix.py tests/test_lagrangian_front_matrix_compatibility.py -q
```

Fresh geographic and Cartesian reference runs were captured before extraction. Each includes the matrix, configuration, all 26 scientific result tables and remaining scalar/summary fields. A frozen source archive and SHA-256 file inventory are retained privately under `.local/reference/`. Existing historical regression fixtures were also migrated without rewriting their numerical tables.

The target is the sibling `lagrangian-fronts/` project. Distribution and CLI: **lagrangian-fronts**. Import package: **lagrangian_fronts**. Software version: **0.1.0**; configuration version remains **1**.

## Files and interfaces

```text
lagrangian-fronts/
├── pyproject.toml
├── MANIFEST.in
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── .gitignore
├── src/lagrangian_fronts/
│   ├── __init__.py, __main__.py, cli.py
│   ├── analysis.py, config.py, workflow.py, io.py
│   ├── trajectories.py, matrix.py, statistics.py, geometry.py
│   ├── cores.py, fronts.py
│   ├── directional_corridors.py, directional_cores.py, directional_fronts.py
│   ├── comparison.py, validation.py, plotting.py
│   ├── plot_flux_cores_hydrography.py
│   └── four existing private scientific kernels
├── tests/                         # migrated tests and unchanged historical fixtures
├── examples/
│   ├── geographic/                # trajectories.csv, config.yaml, load.yaml
│   ├── cartesian/                 # trajectories.csv, config.yaml, load.yaml
│   └── production/                # portable ARGO/drifter matrix-load templates
└── docs/
    ├── INPUT_DATA.md
    ├── METHODOLOGY.md
    ├── CONFIGURATION.md
    └── EXTRACTION_REPORT.md
```

All 23 original Python modules were migrated, retaining their internal names. **19 files are byte-identical** to the source. Changes to the other four are packaging-only: software version, manifest version lookup, delegation of `__main__` to the new `cli.py`, and removal of personal paths from auxiliary plotting documentation. The CLI parser and its single `--config` argument were moved unchanged into `cli.py`. Existing relative internal imports already provided independence; consuming tests now import `lagrangian_fronts`.

The public in-memory API, YAML semantics, flux names, matrix schemas, normalization, stays, segment modes, interpolation/gaps, geographic wrapping, ridge/plateau/connectivity logic, selections, front persistence and output organization are preserved. The matrix is passed to analysis in memory. Software version is sourced from `__version__` for distribution metadata and run manifests.

All six configurations were adapted. Tiny examples use relative input/output paths. Production templates retain every scientific and plotting setting, changing only data/output locations. Original machine-specific production YAMLs remain private under `.local/configs/`; they are excluded from Git and distributions.

## Dependencies, license and citation

Runtime dependencies declared in `pyproject.toml`: NumPy, pandas, PyArrow, xarray, netCDF4, Zarr, PyYAML, SciPy, pyproj, Matplotlib and Cartopy. SciPy supports correlation calculations; PyArrow, netCDF4 and Zarr provide the requested file backends. Other libraries such as Shapely are supplied by their direct dependants. No parent framework utility needed copying.

The initial release retains NumPy 1.x, pandas 2.x and Zarr 2.x compatibility bounds; no exact runtime versions are pinned. Python's minimum remains 3.10, with validation performed on Python 3.12.3/Windows. The `test` extra adds pytest and the standard `build` frontend. The backend is setuptools with a conventional `src/` layout. See the [PyPA metadata guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for the packaging conventions used.

**Remaining kinematicParcels/research runtime dependencies: none.** An AST import audit found only standard-library and declared scientific dependencies. Public package code, examples, docs and tests were checked for personal absolute paths. Git metadata remains optional.

`LICENSE` is a byte-for-byte copy of the parent's MIT license, preserving the 2026 Jacopo Busatto copyright. `CITATION.cff` contains the supported software title, version, author and license and validates against the official CFF 1.2.0 JSON schema; no DOI, ORCID, affiliation or publication citation was invented. There is no configured remote repository or PyPI release.

## Regression and editable installation

Commands from the standalone project, using a fresh virtual environment with `include-system-site-packages = false`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe .local/compare_current.py
```

Installation and `pip check` passed without kinematicParcels or a `PYTHONPATH` adjustment. **189 tests passed, 90 warnings in 69.56 s**: all 184 migrated module tests plus five packaging tests. The other 47 tests in the 231-test parent baseline belong to the parent matrix builder (43) and cross-project comparison (4). They are not standalone runtime requirements. No migrated scientific assertion was removed or weakened.

Warnings comprise the existing netCDF NumPy binary-size warning and empty-slice mean warnings exposed by the independently resolved dependency stack. Scientific outputs still compare exactly; these warnings were not suppressed or used to relax assertions.

Both fresh geographic and Cartesian comparisons passed **exactly across all 26 scientific tables and summaries**, including validation and directional plateau thinning. Comparison uses the same Parquet serialization boundary to preserve historical object/bool inference. Numeric tolerance is zero (`check_exact=True`); row/column ordering and dtypes are checked. No scientific columns are ignored. The in-memory comparisons contain no timestamps, Git commits or package-version metadata to exclude.

The installed test suite runs both command entry points from external working directories. Its geographic and Cartesian console examples each run compute and load modes with validation/debug outputs enabled, producing **21 analysis tables and 8 figures per run**. Every load-mode analysis table matches the corresponding compute run exactly. YAML-relative paths, version metadata, detected fronts and absence of a duplicated loaded matrix are checked.

## Distribution and source handoff

`python -m build` successfully produced `dist/lagrangian_fronts-0.1.0.tar.gz` and `dist/lagrangian_fronts-0.1.0-py3-none-any.whl`, using an isolated setuptools build environment. The wheel contains all 24 package modules, entry-point/distribution metadata and the unchanged MIT license. The source archive additionally includes the examples, docs and complete tests/fixtures. Neither includes private configs, external datasets, generated outputs, environments or the parent framework.

A second virtual environment was created outside both repositories, again without system site packages. The built wheel was installed normally with its test extra, and the source archive was extracted outside the repositories. From an unrelated working directory, these checks passed:

```text
python -c "import lagrangian_fronts; print(lagrangian_fronts.__version__)"
lagrangian-fronts --help
python -m lagrangian_fronts --help
python -m pip check
python -m pytest <extracted-source-archive>/tests -q
```

The import resolved to the wheel environment's `site-packages`, version **0.1.0**; neither `kinematicparcels` nor `research` was importable. Both entry points expose the same single YAML option. The full wheel-backed suite passed **189 tests, 90 warnings in 84.58 s**, including geographic/Cartesian compute/load execution through the installed console script. No `PYTHONPATH` was supplied.

The resolved fresh environment used NumPy 1.26.4, pandas 2.3.3, PyArrow 25.0.1, xarray 2026.7.0, netCDF4 1.7.4, Zarr 2.18.7, PyYAML 6.0.3, SciPy 1.17.1, pyproj 3.8.0, Matplotlib 3.11.1 and Cartopy 0.25.0. These versions differ from several parent-environment versions while preserving the exact scientific regression results.

All standalone acceptance checks completed before source cleanup. The standalone project was initialized as a local Git repository on `main` and committed as `a7dfbed` (`Extract validated Lagrangian fronts as standalone 0.1.0`), establishing the authoritative source before the handoff.

The old implementation and migrated tests have been removed from the parent's active source tree. `research/lagrangian_fronts/` now contains only a README migration pointer, and the parent README links to it. The recursive deletion command was blocked by the execution policy, so the handoff used reversible moves: the retired module, tests and original generated example outputs are preserved under the standalone repository's ignored `.local/retired-source/`. This is a private recovery archive, excluded from Git, package discovery and distributions. The original production YAMLs are also available under `.local/configs/`, and the pre-extraction ZIP and hash inventory remain under `.local/reference/`. No second scientific implementation is maintained.

The parent's optional cross-project oracle now imports the installed `lagrangian_fronts` package, retaining all four comparisons and exact assertions. It skips only when that independent package is absent. No parent runtime dependency was added. After handoff, from the parent repository:

```powershell
python -m pytest tests/test_gridded_transition_matrix.py tests/test_lagrangian_front_matrix_compatibility.py -q
```

Result without the standalone package installed: **43 passed, 1 skipped, 1 warning in 1.84 s**. To also verify the four cross-project comparisons, a separate integration environment inherited the existing parent dependencies and installed the standalone wheel without changing the parent environment:

```powershell
python -m venv --system-site-packages ../lagrangian-fronts/.local/parent-oracle-env
& ../lagrangian-fronts/.local/parent-oracle-env/Scripts/python.exe -m pip install --no-deps ../lagrangian-fronts/dist/lagrangian_fronts-0.1.0-py3-none-any.whl
& ../lagrangian-fronts/.local/parent-oracle-env/Scripts/python.exe -m pytest tests/test_gridded_transition_matrix.py tests/test_lagrangian_front_matrix_compatibility.py -q
```

Result: **47 passed, 1 warning in 2.04 s**. This parent interoperability check is separate from the two clean standalone environments described above. The installed wheel was also rechecked after handoff from an external directory: version/import and module CLI still work, with neither parent package importable.

The source archive and wheel were rebuilt after this report was finalized; package source bytes remain identical to the wheel that passed the full installed test suite. Public source, examples and documentation contain no personal data paths. The parent repository's unrelated working changes and Git history were preserved; its handoff changes remain uncommitted for the maintainer to review.

## Remaining maintainer steps and limits

Local extraction and validation are complete. Before publishing, the maintainer can create a remote repository, replace the README's `<repository-url>` placeholder, add the remote and push the local history. No remote was created and nothing was pushed or published to PyPI. A paper citation, DOI or ORCID can be added only when available. To share the project now, use the source archive (includes examples, docs and tests) or wheel (package and CLI); private `.local` content is excluded.

Scientific scope is unchanged: input trajectories must already be prepared, selected observations are materialized in memory, and specialized ragged layouts/nonstandard calendars require upstream conversion. External production matrices are not bundled; their portable templates retain scientific settings and require local input paths. Python 3.12.3 on Windows was exercised; other supported Python versions and platforms were not tested in this extraction. No QC, grid inference, CRS conversion, new front methods or other scientific changes were introduced.
